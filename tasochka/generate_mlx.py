"""BPE-aware generation + CLI chat.

Features that matter for a small model:
  - repetition penalty (kills the "Изучите задачи. Изучите задачи." loops)
  - no-repeat n-gram blocking (hard barrier against verbatim loops)
  - nucleus (top-p) sampling alongside top-k
  - UTF-8-safe streaming (no more `���` characters mid-output)
  - chat-template prompt wrapping with auto-stop at </assistant>
  - KV cache: the prompt is processed once, then one token per step; the
    shared loop (sampling, n-grams, decoding, stops) lives in sampling.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from .model_mlx import MiniGPT, ModelConfig, load_checkpoint, make_cache
from .sampling import (  # noqa: F401  (re-exported for callers)
    CHAT_STOPS,
    DEFAULT_NO_REPEAT_NGRAM,
    DEFAULT_REP_PENALTY,
    DEFAULT_REP_WINDOW,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    build_chat_prompt,
    stream_generation,
)
from .tokenizer import BPETokenizer

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "tasochka"

# MLX keeps freed GPU buffers in a cache for reuse; without a cap that cache
# can grow until macOS swaps itself to a halt. 1 GB is plenty for inference.
MLX_CACHE_LIMIT_BYTES = 1 << 30


def _mx_memory_fn(name: str):
    """mx.<name> on new MLX, mx.metal.<name> on older releases."""
    for module in (mx, getattr(mx, "metal", None)):
        fn = getattr(module, name, None) if module is not None else None
        if fn is not None:
            return fn
    return None


def limit_mlx_cache(limit_bytes: int = MLX_CACHE_LIMIT_BYTES) -> None:
    fn = _mx_memory_fn("set_cache_limit")
    if fn is not None:
        fn(limit_bytes)


def clear_mlx_cache() -> None:
    fn = _mx_memory_fn("clear_cache")
    if fn is not None:
        fn()


def load(checkpoint_dir: Path) -> Tuple[MiniGPT, BPETokenizer, ModelConfig]:
    limit_mlx_cache()
    model, cfg = load_checkpoint(checkpoint_dir)
    tokenizer = BPETokenizer.load(checkpoint_dir / "tokenizer.json")
    model.eval()
    return model, tokenizer, cfg


class _CachedStepper:
    """next-token logits with a KV cache: the prompt is processed in one pass,
    then each call runs the model on a single new token.

    When the sequence outgrows the context window, the cache is rebuilt from
    the last 3/4 of the window (positions restart at 0, as in training); the
    next rebuild comes only after another 1/4 window of tokens, so the cost of
    recomputing stays small on long answers.
    """

    def __init__(self, model: MiniGPT, cfg: ModelConfig):
        self.model = model
        self.ctx = cfg.context_length
        self.cache = make_cache(cfg)
        self.start = 0      # index in ids of the first cached token
        self.n_cached = 0   # cached tokens; 0 means "prefill next"

    def __call__(self, ids: List[int]) -> np.ndarray:
        new = ids[self.start + self.n_cached:]
        if self.n_cached == 0 or self.n_cached + len(new) > self.ctx:
            keep = self.ctx if self.n_cached == 0 else self.ctx - self.ctx // 4
            keep = min(len(ids), keep)
            for layer in self.cache:
                layer.offset = 0  # reuse the same buffers
            self.start = len(ids) - keep
            new = ids[self.start:]
        x = mx.array(np.array(new, dtype=np.int32).reshape(1, -1))
        logits = self.model(x, cache=self.cache, last_only=True)
        self.n_cached = len(ids) - self.start
        # fp32 cast first: numpy can't read bf16 buffers directly
        return np.array(logits[0, -1].astype(mx.float32))


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
    try:
        yield from stream_generation(
            _CachedStepper(model, cfg), tokenizer, prompt,
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            top_p=top_p, rep_penalty=rep_penalty, rep_window=rep_window,
            no_repeat_ngram=no_repeat_ngram, stop_strings=stop_strings,
        )
    finally:
        # Also runs when the caller stops early (stop string, client gone).
        clear_mlx_cache()


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
        print("Введите Запрос. Пустая строка или exit — выход.")
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
        prompt = build_chat_prompt(args.prompt or "Hello!")
        stops = args.stop if args.stop else CHAT_STOPS

    for piece in generate_stream(model, tokenizer, cfg, prompt,
                                  stop_strings=stops, **gen_kwargs):
        print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
