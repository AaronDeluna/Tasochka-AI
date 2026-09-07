"""BPE-aware generation + CLI chat.

Features that matter for a small model:
  - repetition penalty (kills the "Изучите задачи. Изучите задачи." loops)
  - no-repeat n-gram blocking (hard barrier against verbatim loops)
  - nucleus (top-p) sampling alongside top-k
  - UTF-8-safe streaming (no more `���` characters mid-output)
  - chat-template prompt wrapping with auto-stop at </assistant>
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Set, Tuple

import mlx.core as mx
import numpy as np

from .model_mlx import MiniGPT, ModelConfig, load_checkpoint
from .tokenizer import BPETokenizer

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "tasochka"

CHAT_STOPS = ["</assistant>", "<user>", "<|endoftext|>", "</s>"]

# Sampling defaults tuned for the trained small model
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.92
DEFAULT_REP_PENALTY = 1.18
DEFAULT_NO_REPEAT_NGRAM = 3
DEFAULT_REP_WINDOW = 128


def build_chat_prompt(user_message: str) -> str:
    return f"<user>\n{user_message.strip()}\n</user>\n<assistant>\n"


def load(checkpoint_dir: Path) -> Tuple[MiniGPT, BPETokenizer, ModelConfig]:
    model, cfg = load_checkpoint(checkpoint_dir)
    tokenizer = BPETokenizer.load(checkpoint_dir / "tokenizer.json")
    model.eval()
    return model, tokenizer, cfg


def _banned_ngram_tokens(generated_ids: List[int], ngram_size: int) -> Set[int]:
    """Tokens that, if generated next, would complete a verbatim n-gram repeat."""
    if ngram_size <= 0 or len(generated_ids) < ngram_size:
        return set()
    prefix = tuple(generated_ids[-(ngram_size - 1):])
    banned: Set[int] = set()
    for i in range(len(generated_ids) - ngram_size + 1):
        ngram = tuple(generated_ids[i:i + ngram_size])
        if ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return banned


def _sample_token(
    logits: mx.array,
    temperature: float,
    top_k: int,
    top_p: Optional[float],
    recent_ids: Sequence[int],
    rep_penalty: float,
    banned_ids: Optional[Set[int]] = None,
) -> int:
    mx.eval(logits)
    logits_np: np.ndarray = np.array(logits).astype(np.float64)

    # 1. Repetition penalty: pull recent tokens toward zero
    if rep_penalty != 1.0 and recent_ids:
        unique = set(int(t) for t in recent_ids)
        for tid in unique:
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
    if top_k > 0 and top_k < len(logits_np):
        kth = np.partition(logits_np, -top_k)[-top_k]
        logits_np = np.where(logits_np >= kth, logits_np, -1e9)

    # 5. Softmax
    logits_np = logits_np - logits_np.max()
    probs = np.exp(logits_np)
    s = probs.sum()
    if s <= 0 or not np.isfinite(s):
        # All logits filtered out — fall back to argmax of original
        return int(np.argmax(np.array(logits)))
    probs = probs / s

    # 6. Top-p (nucleus)
    if top_p is not None and 0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        sorted_probs = probs[order]
        cumsum = np.cumsum(sorted_probs)
        cutoff = int(np.searchsorted(cumsum, top_p)) + 1
        keep_idx = order[:cutoff]
        new_probs = np.zeros_like(probs)
        new_probs[keep_idx] = probs[keep_idx]
        s2 = new_probs.sum()
        if s2 > 0:
            probs = new_probs / s2

    return int(np.random.choice(len(probs), p=probs))


def _find_earliest_stop(text: str, stops: Sequence[str]) -> Optional[int]:
    earliest = None
    for stop in stops:
        idx = text.find(stop)
        if idx >= 0 and (earliest is None or idx < earliest):
            earliest = idx
    return earliest


def generate_stream(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    cfg: ModelConfig,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
    top_p: Optional[float] = DEFAULT_TOP_P,
    rep_penalty: float = DEFAULT_REP_PENALTY,
    rep_window: int = DEFAULT_REP_WINDOW,
    no_repeat_ngram: int = DEFAULT_NO_REPEAT_NGRAM,
    stop_strings: Optional[Sequence[str]] = None,
) -> Iterator[str]:
    """Yield decoded chunks. Holds back trailing U+FFFD until the multi-byte sequence completes."""
    prompt_ids: List[int] = tokenizer.encode(prompt).tolist() if prompt else []
    if not prompt_ids:
        prompt_ids = [0]
    ids: List[int] = list(prompt_ids)
    generated_ids: List[int] = []  # tokens produced after the prompt

    initial_text = tokenizer.decode(ids)
    yielded_len = len(initial_text)
    generated_text = ""  # only what's been generated, used for stop matching

    for _ in range(max_new_tokens):
        context = ids[-cfg.context_length:]
        x = mx.array(np.array(context, dtype=np.int32).reshape(1, -1))
        logits = model(x)
        last = logits[0, -1]

        recent = generated_ids[-rep_window:] if rep_window > 0 else []
        banned = _banned_ngram_tokens(generated_ids, no_repeat_ngram) if no_repeat_ngram > 1 else None

        next_id = _sample_token(
            last,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            recent_ids=recent,
            rep_penalty=rep_penalty,
            banned_ids=banned,
        )
        ids.append(next_id)
        generated_ids.append(next_id)

        # Re-decode full sequence; rightmost incomplete byte shows as U+FFFD —
        # treat those as "not yet ready" and hold them for the next iteration.
        full_text = tokenizer.decode(ids)
        safe_end = len(full_text)
        while safe_end > yielded_len and full_text[safe_end - 1] == "�":
            safe_end -= 1
        if safe_end <= yielded_len:
            continue

        delta = full_text[yielded_len:safe_end]
        generated_text += delta

        if stop_strings:
            stop_at = _find_earliest_stop(generated_text, stop_strings)
            if stop_at is not None:
                already = len(generated_text) - len(delta)
                keep = max(0, stop_at - already)
                if keep > 0:
                    yield delta[:keep]
                return

        yield delta
        yielded_len = safe_end


def generate(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    cfg: ModelConfig,
    prompt: str,
    **kwargs,
) -> str:
    return "".join(generate_stream(model, tokenizer, cfg, prompt, **kwargs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-new", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--rep-penalty", type=float, default=DEFAULT_REP_PENALTY,
                       help="Repetition penalty (1.0 = off, 1.1–1.3 typical)")
    parser.add_argument("--rep-window", type=int, default=DEFAULT_REP_WINDOW,
                       help="How many recent tokens the penalty applies to")
    parser.add_argument("--no-repeat-ngram", type=int, default=DEFAULT_NO_REPEAT_NGRAM,
                       help="Block exact n-gram repeats; 0 to disable")
    parser.add_argument("--chat", action="store_true",
                       help="Interactive REPL: wraps input in chat template, auto-stops on </assistant>")
    parser.add_argument("--raw", action="store_true",
                       help="Don't wrap prompt in chat tags and don't auto-stop")
    parser.add_argument("--stop", action="append", default=None,
                       help="Stop string; can be passed multiple times.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    model, tokenizer, cfg = load(args.checkpoint)

    gen_kwargs = dict(
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p if args.top_p > 0 else None,
        rep_penalty=args.rep_penalty,
        rep_window=args.rep_window,
        no_repeat_ngram=args.no_repeat_ngram,
    )

    if args.chat:
        print("Введите prompt. Пустая строка или exit — выход.")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line or line.lower() == "exit":
                break
            prompt = build_chat_prompt(line)
            stops = args.stop if args.stop else CHAT_STOPS
            for piece in generate_stream(model, tokenizer, cfg, prompt,
                                          stop_strings=stops, **gen_kwargs):
                print(piece, end="", flush=True)
            print()
        return

    if args.raw:
        prompt = args.prompt or "- "
        stops = args.stop
        print(prompt, end="", flush=True)
    else:
        prompt = build_chat_prompt(args.prompt or "Привет")
        stops = args.stop if args.stop else CHAT_STOPS

    for piece in generate_stream(model, tokenizer, cfg, prompt,
                                  stop_strings=stops, **gen_kwargs):
        print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
