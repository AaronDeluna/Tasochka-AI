"""Training script: BPE tokenizer + LLaMA-style mini transformer.

Steps:
    1. Load and clean a JSONL file, TXT file, or directory with TXT files.
    2. Train (or load) BPE tokenizer.
    3. Encode the corpus once; cache tokens to disk for fast resume.
    4. AdamW + global L2 grad clipping + val loss.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .data import (
    TokenDataset,
    corpus_signature,
    encode_corpus_to_file,
    iter_cleaned_chunks,
    split_train_val_ids,
)
from .model_mlx import (
    MiniGPT,
    ModelConfig,
    load_checkpoint,
    param_count,
    save_checkpoint,
)
from .tokenizer import BPETokenizer

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "russian-train-dataset"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "tasochka"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-dataset-chars", type=int, default=30_000_000)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=896)
    parser.add_argument("--num-layers", type=int, default=20)
    parser.add_argument("--num-heads", type=int, default=14)
    parser.add_argument("--feed-forward-dim", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                       help="Cosine decay floor as fraction of --lr (final LR = lr * ratio).")
    return parser.parse_args()


def lr_schedule(step: int, max_steps: int, warmup_steps: int,
                base_lr: float, min_ratio: float) -> float:
    if step <= warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def global_grad_norm(grads) -> mx.array:
    total = mx.array(0.0)
    flat: list = []
    _collect_arrays(grads, flat)
    for g in flat:
        total = total + (g * g).sum()
    return mx.sqrt(total)


def _collect_arrays(obj, out: list) -> None:
    if isinstance(obj, mx.array):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_arrays(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_arrays(v, out)


def _scale_grads(grads, scale: mx.array):
    if isinstance(grads, mx.array):
        return grads * scale
    if isinstance(grads, dict):
        return {k: _scale_grads(v, scale) for k, v in grads.items()}
    if isinstance(grads, list):
        return [_scale_grads(v, scale) for v in grads]
    if isinstance(grads, tuple):
        return tuple(_scale_grads(v, scale) for v in grads)
    return grads


def get_or_train_tokenizer(data_path: Path, checkpoint_dir: Path, vocab_size: int,
                           bpe_max_chars: int) -> BPETokenizer:
    path = checkpoint_dir / "tokenizer.json"
    if path.exists():
        print(f"loading tokenizer from {path}")
        return BPETokenizer.load(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"training BPE tokenizer (vocab={vocab_size}, on {bpe_max_chars / 1e6:.0f} MB)...")
    start = time.time()
    # Subsample the corpus for BPE training — we don't need all 3.5 GB
    # to learn 8000 merges, ~200 MB is more than enough.
    chunk_iter = iter_cleaned_chunks(data_path, bpe_max_chars)
    tok = BPETokenizer.train(chunk_iter, vocab_size=vocab_size)
    tok.save(path)
    print(f"tokenizer trained in {format_duration(time.time() - start)}, vocab={tok.vocab_size}")
    return tok


def get_or_encode_ids(data_path: Path, tokenizer: BPETokenizer, checkpoint_dir: Path,
                     max_chars: int, source_signature: str) -> np.memmap:
    """Encode corpus to a memmap-backed int32 file. Re-uses cache if signature matches."""
    ids_path = checkpoint_dir / "corpus_ids.bin"
    meta_path = checkpoint_dir / "corpus_ids.meta"
    meta_value = f"{max_chars}:{tokenizer.vocab_size}:{source_signature}"
    if ids_path.exists() and meta_path.exists():
        meta = meta_path.read_text(encoding="utf-8").strip()
        if meta == meta_value:
            n_tokens = ids_path.stat().st_size // 4
            print(f"loading cached token ids from {ids_path} ({n_tokens / 1e6:.1f}M tokens)")
            return np.memmap(ids_path, dtype=np.int32, mode="r", shape=(n_tokens,))
    print(f"encoding corpus (streaming, max {max_chars / 1e6:.0f} MB)...")
    start = time.time()
    cleaned_chars, n_tokens = encode_corpus_to_file(
        data_path, tokenizer, max_chars, ids_path
    )
    meta_path.write_text(meta_value, encoding="utf-8")
    print(f"encoded in {format_duration(time.time() - start)}: "
          f"{cleaned_chars / 1e6:.1f}M cleaned chars → {n_tokens / 1e6:.1f}M tokens "
          f"(compression {cleaned_chars / max(1, n_tokens):.2f} chars/token)")
    return np.memmap(ids_path, dtype=np.int32, mode="r", shape=(n_tokens,))


def main() -> None:
    args = parse_args()
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    args.checkpoint.mkdir(parents=True, exist_ok=True)

    print(f"loading corpus from {args.data}")
    sig = corpus_signature(args.data)

    bpe_cap = min(args.max_dataset_chars, 200_000_000)
    tokenizer = get_or_train_tokenizer(args.data, args.checkpoint, args.vocab_size, bpe_cap)
    ids = get_or_encode_ids(args.data, tokenizer, args.checkpoint, args.max_dataset_chars, sig)

    weights_path = args.checkpoint / "weights.safetensors"
    config_path = args.checkpoint / "config.json"

    if weights_path.exists() and config_path.exists():
        print(f"resuming from {args.checkpoint}")
        model, cfg = load_checkpoint(args.checkpoint)
    else:
        print("starting new model")
        cfg = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=args.context_length,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            feed_forward_dim=args.feed_forward_dim,
        )
        model = MiniGPT(cfg)
        mx.eval(model.parameters())

    print(f"vocab={tokenizer.vocab_size} params={param_count(model):,} "
          f"ctx={cfg.context_length} layers={cfg.num_layers} dim={cfg.embedding_dim}")

    train_ids, val_ids = split_train_val_ids(ids, args.val_fraction)
    train_set = TokenDataset.from_ids(train_ids, args.seed)
    val_set = TokenDataset.from_ids(val_ids, args.seed + 1)
    print(f"train tokens={len(train_ids):,} val tokens={len(val_ids):,}")

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: m.loss(x, y))

    def step_fn(idx: mx.array, targets: mx.array, lr: mx.array):
        loss, grads = loss_and_grad(model, idx, targets)
        norm = global_grad_norm(grads)
        scale = mx.minimum(mx.array(1.0), args.grad_clip / (norm + 1e-6))
        grads = _scale_grads(grads, scale)
        optimizer.learning_rate = lr
        optimizer.update(model, grads)
        return loss, norm

    # sanity probe
    x0, y0 = train_set.next_batch(args.batch_size, cfg.context_length)
    loss0 = model.loss(mx.array(x0), mx.array(y0))
    mx.eval(loss0)
    print(f"init forward loss={loss0.item():.4f} (random baseline ~{math.log(tokenizer.vocab_size):.2f})")

    started = time.time()
    last_log = started
    last_log_step = 0
    recent_loss = 0.0
    recent_count = 0

    for step in range(1, args.max_steps + 1):
        x, y = train_set.next_batch(args.batch_size, cfg.context_length)
        lr = lr_schedule(step, args.max_steps, args.warmup_steps, args.lr, args.min_lr_ratio)
        loss, norm = step_fn(mx.array(x), mx.array(y), mx.array(lr))
        mx.eval(loss, norm, model.parameters(), optimizer.state)
        loss_v = loss.item()
        recent_loss += loss_v
        recent_count += 1

        if step == 1 or step % args.log_every == 0:
            now = time.time()
            avg = recent_loss / max(1, recent_count)
            ms_per_step = (now - last_log) * 1000.0 / max(1, step - last_log_step)
            eta = (now - started) / step * (args.max_steps - step)
            print(
                f"step {step}/{args.max_steps} ({100.0 * step / args.max_steps:.1f}%) "
                f"loss {loss_v:.4f} (avg {avg:.4f}) gnorm {float(norm):.2f} lr {lr:.2e} "
                f"{ms_per_step:.0f} ms/step ETA {format_duration(eta)}"
            )
            last_log = now
            last_log_step = step
            recent_loss = 0.0
            recent_count = 0

        if args.val_every and step % args.val_every == 0:
            val_loss = evaluate(model, val_set, args.batch_size, cfg.context_length, args.val_batches)
            print(f"[val] step {step} loss {val_loss:.4f}")

        if args.checkpoint_every and step % args.checkpoint_every == 0:
            save_checkpoint(args.checkpoint, model, cfg)
            print(f"[checkpoint] step {step} saved to {args.checkpoint}")

    save_checkpoint(args.checkpoint, model, cfg)
    print(f"saved final model to {args.checkpoint}")


def evaluate(model: MiniGPT, dataset: TokenDataset, batch_size: int, ctx: int, batches: int) -> float:
    total = 0.0
    n = 0
    for _ in range(batches):
        x, y = dataset.next_batch(batch_size, ctx)
        loss = model.loss(mx.array(x), mx.array(y))
        mx.eval(loss)
        total += loss.item()
        n += 1
    return total / max(1, n)


if __name__ == "__main__":
    main()
