"""Backend-independent part of generation: sampling, n-gram blocking,
incremental UTF-8-safe decoding and stop strings.

`generate_mlx.py` and `generate_torch.py` only supply a function that returns
next-token logits as a numpy vector; the loop in `stream_generation` is shared,
so both backends behave identically token for token.

Every step here costs O(1) in the length of the answer (apart from the model):
n-grams are indexed as they appear, only the not-yet-printed tail is decoded,
and stop strings are searched only in text that has not been printed.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

from .tokenizer import BPETokenizer

CHAT_STOPS = ["</assistant>", "<user>", "<|endoftext|>", "</s>"]

# Sampling defaults tuned for the trained small model
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.92
DEFAULT_REP_PENALTY = 1.18
DEFAULT_NO_REPEAT_NGRAM = 3
DEFAULT_REP_WINDOW = 128

# A token run that still decodes to a trailing U+FFFD after this many tokens is
# not an unfinished UTF-8 character but genuinely invalid bytes: print it.
_MAX_PENDING_TOKENS = 16


def build_chat_prompt(user_message: str) -> str:
    return f"<user>\n{user_message.strip()}\n</user>\n<assistant>\n"


def sample_token(
    logits: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: Optional[float],
    recent_ids: Sequence[int],
    rep_penalty: float,
    banned_ids: Optional[Set[int]] = None,
) -> int:
    """Pick the next token from a 1-D vector of logits (any float dtype)."""
    original = logits
    logits_np = np.asarray(logits, dtype=np.float64).copy()

    # 1. Repetition penalty: pull recent tokens toward zero
    if rep_penalty != 1.0 and recent_ids:
        for tid in set(int(t) for t in recent_ids):
            if 0 <= tid < len(logits_np):
                logits_np[tid] = (
                    logits_np[tid] / rep_penalty
                    if logits_np[tid] > 0
                    else logits_np[tid] * rep_penalty
                )

    # 2. Block tokens that would repeat an n-gram verbatim
    if banned_ids:
        for tid in banned_ids:
            if 0 <= tid < len(logits_np):
                logits_np[tid] = -1e9

    # 3. Temperature
    logits_np = logits_np / max(temperature, 1e-6)

    # 4. Top-k
    if 0 < top_k < len(logits_np):
        kth = np.partition(logits_np, -top_k)[-top_k]
        logits_np = np.where(logits_np >= kth, logits_np, -1e9)

    # 5. Softmax
    logits_np = logits_np - logits_np.max()
    probs = np.exp(logits_np)
    s = probs.sum()
    if s <= 0 or not np.isfinite(s):
        # All logits filtered out — fall back to argmax of original
        return int(np.argmax(original))
    probs = probs / s

    # 6. Top-p (nucleus)
    if top_p is not None and 0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[order])
        cutoff = int(np.searchsorted(cumsum, top_p)) + 1
        keep_idx = order[:cutoff]
        new_probs = np.zeros_like(probs)
        new_probs[keep_idx] = probs[keep_idx]
        s2 = new_probs.sum()
        if s2 > 0:
            probs = new_probs / s2

    return int(np.random.choice(len(probs), p=probs))


class NgramBlocker:
    """Tokens that would complete a verbatim repeat of an earlier n-gram.

    Keeps an index {(n-1)-token prefix -> next tokens seen after it}, updated
    once per token, instead of rescanning the whole answer every step.
    """

    def __init__(self, ngram_size: int):
        self.n = ngram_size
        self.ids: List[int] = []
        self.index: Dict[Tuple[int, ...], Set[int]] = {}

    def add(self, token: int) -> None:
        self.ids.append(token)
        if len(self.ids) >= self.n:
            prefix = tuple(self.ids[-self.n:-1])
            self.index.setdefault(prefix, set()).add(token)

    def banned(self) -> Set[int]:
        if len(self.ids) < self.n:
            return set()
        return self.index.get(tuple(self.ids[-(self.n - 1):]), set())


class IncrementalDecoder:
    """Turns tokens into text without re-decoding the whole answer.

    A multi-byte UTF-8 character can be split across tokens; until it is
    complete it decodes to a trailing U+FFFD. Only the tokens after the last
    fully decoded point are kept and re-decoded, so the cost per step is tiny.
    """

    def __init__(self, tokenizer: BPETokenizer):
        self.tokenizer = tokenizer
        self.pending: List[int] = []
        self.pending_emitted = 0  # chars of decode(pending) already returned

    def push(self, token: int) -> str:
        self.pending.append(token)
        text = self.tokenizer.decode(self.pending)
        safe_end = len(text)
        if len(self.pending) < _MAX_PENDING_TOKENS:
            while safe_end > self.pending_emitted and text[safe_end - 1] == "�":
                safe_end -= 1
        delta = text[self.pending_emitted:safe_end] if safe_end > self.pending_emitted else ""
        if safe_end == len(text):
            # Everything decoded cleanly: later tokens start from a clean slate.
            self.pending = []
            self.pending_emitted = 0
        else:
            self.pending_emitted = max(self.pending_emitted, safe_end)
        return delta


def _find_earliest_stop(text: str, stops: Sequence[str], start: int = 0) -> Optional[int]:
    earliest = None
    for stop in stops:
        idx = text.find(stop, start)
        if idx >= 0 and (earliest is None or idx < earliest):
            earliest = idx
    return earliest


def _partial_stop_len(text: str, stops: Sequence[str]) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of a stop.

    Tokens arrive piecewise ("</", "assistant", ">"), so text that might be the
    start of a stop string is held back until the next token decides it.
    """
    best = 0
    for stop in stops:
        for n in range(min(len(stop) - 1, len(text)), best, -1):
            if text.endswith(stop[:n]):
                best = n
                break
    return best


def stream_generation(
    next_logits: Callable[[List[int]], np.ndarray],
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: Optional[float],
    rep_penalty: float,
    rep_window: int,
    no_repeat_ngram: int,
    stop_strings: Optional[Sequence[str]],
) -> Iterator[str]:
    """Yield decoded chunks of the answer.

    `next_logits(ids)` gets the whole sequence so far (prompt + generated) and
    returns logits for the next token; each call has exactly one more token
    than the previous one, which lets the backend keep a KV cache.
    """
    prompt_ids: List[int] = tokenizer.encode(prompt).tolist() if prompt else []
    if not prompt_ids:
        prompt_ids = [0]
    ids: List[int] = list(prompt_ids)
    generated_ids: List[int] = []  # tokens produced after the prompt
    blocker = NgramBlocker(no_repeat_ngram) if no_repeat_ngram > 1 else None
    decoder = IncrementalDecoder(tokenizer)

    generated_text = ""  # only what's been generated, used for stop matching
    emitted = 0  # how much of generated_text has been yielded

    for _ in range(max_new_tokens):
        logits = next_logits(ids)
        recent = generated_ids[-rep_window:] if rep_window > 0 else []
        next_id = sample_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            recent_ids=recent,
            rep_penalty=rep_penalty,
            banned_ids=blocker.banned() if blocker else None,
        )
        ids.append(next_id)
        generated_ids.append(next_id)
        if blocker:
            blocker.add(next_id)

        delta = decoder.push(next_id)
        if not delta:
            continue
        generated_text += delta

        if stop_strings:
            # Text before `emitted` was already checked and contains no stop,
            # and a stop can't start there either: its beginning would have
            # been held back as a partial match.
            stop_at = _find_earliest_stop(generated_text, stop_strings, emitted)
            if stop_at is not None:
                if stop_at > emitted:
                    yield generated_text[emitted:stop_at]
                return
            ready = len(generated_text) - _partial_stop_len(generated_text, stop_strings)
        else:
            ready = len(generated_text)
        if ready > emitted:
            yield generated_text[emitted:ready]
            emitted = ready

    # max_new_tokens hit: whatever is still held back is the start of a stop
    # string ("</assist"), so drop it rather than print a half tag.
    if not stop_strings and len(generated_text) > emitted:
        yield generated_text[emitted:]
