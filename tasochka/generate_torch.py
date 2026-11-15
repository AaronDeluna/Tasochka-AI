"""PyTorch generation — same UX as generate_mlx.py.

Features mirrored from MLX version:
  - repetition penalty
  - no-repeat n-gram blocking
  - nucleus (top-p) + top-k sampling
  - UTF-8-safe streaming (skip trailing replacement char until token completes)
  - chat-template prompt wrapping with auto-stop
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .model_torch import MiniGPT, ModelConfig, load_checkpoint, make_cache, pick_device
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


def load(checkpoint_dir: Path, device: Optional[str] = None
        ) -> Tuple[MiniGPT, BPETokenizer, ModelConfig]:
    dev = pick_device(device)
    model, cfg = load_checkpoint(checkpoint_dir, dev)
    tokenizer = BPETokenizer.load(checkpoint_dir / "tokenizer.json")
    model.eval()
    return model, tokenizer, cfg


class _CachedStepper:
    """next-token logits with a KV cache; same windowing as generate_mlx:
    on overflow the cache is rebuilt from the last 3/4 of the context."""

    def __init__(self, model: MiniGPT, cfg: ModelConfig):
        self.model = model
        self.device = next(model.parameters()).device
        self.ctx = cfg.context_length
        self.cache = make_cache(cfg)
        self.start = 0
        self.n_cached = 0

    @torch.inference_mode()
    def __call__(self, ids: List[int]) -> np.ndarray:
        new = ids[self.start + self.n_cached:]
        if self.n_cached == 0 or self.n_cached + len(new) > self.ctx:
            keep = self.ctx if self.n_cached == 0 else self.ctx - self.ctx // 4
            keep = min(len(ids), keep)
            for layer in self.cache:
                layer.offset = 0
            self.start = len(ids) - keep
            new = ids[self.start:]
        x = torch.tensor([new], dtype=torch.long, device=self.device)
        logits = self.model(x, cache=self.cache, last_only=True)
        self.n_cached = len(ids) - self.start
        return logits[0, -1].float().cpu().numpy()


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
    yield from stream_generation(
        _CachedStepper(model, cfg), tokenizer, prompt,
        max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
        top_p=top_p, rep_penalty=rep_penalty, rep_window=rep_window,
        no_repeat_ngram=no_repeat_ngram, stop_strings=stop_strings,
    )


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
    parser.add_argument("--rep-penalty", type=float, default=DEFAULT_REP_PENALTY)
    parser.add_argument("--rep-window", type=int, default=DEFAULT_REP_WINDOW)
    parser.add_argument("--no-repeat-ngram", type=int, default=DEFAULT_NO_REPEAT_NGRAM)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--stop", action="append", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    model, tokenizer, cfg = load(args.checkpoint, args.device)

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
        prompt = build_chat_prompt(args.prompt or "Hello!")
        stops = args.stop if args.stop else CHAT_STOPS

    for piece in generate_stream(model, tokenizer, cfg, prompt,
                                  stop_strings=stops, **gen_kwargs):
        print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
