"""PyTorch training loop. Same CLI as train_mlx.py.

Device auto-detect: CUDA on NVIDIA server, MPS on Mac, CPU fallback.
On CUDA: BF16 mixed precision via torch.amp for ~2× speedup with no quality loss.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import (
    TokenDataset,
    corpus_signature,
    encode_corpus_to_file,
    iter_cleaned_chunks,
    split_train_val_ids,
)
from .model_torch import (
    MiniGPT,
    ModelConfig,
    init_weights,
    load_checkpoint,
    param_count,
    pick_device,
    save_checkpoint,
)
from .tokenizer import BPETokenizer

DEFAULT_DATA = Path(__file__).resolve().parent.parent.parent / "russian-train-dataset"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "tasochka"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
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
    parser.add_argument("--num-kv-heads", type=int, default=0,
                       help="GQA: KV-голов меньше, чем Q-голов (0 = столько же, обычный MHA).")
    parser.add_argument("--feed-forward-dim", type=int, default=2400)
    parser.add_argument("--tie-embeddings", action="store_true",
                       help="LM-голова использует матрицу эмбеддингов (экономит vocab*dim параметров).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                       help="Cosine decay floor as fraction of --lr.")
    parser.add_argument("--device", type=str, default=None,
                       help="Force device: cuda, mps, cpu. Default: auto-detect.")
    parser.add_argument("--bf16", action="store_true",
                       help="Use bfloat16 mixed precision on CUDA (recommended).")
    parser.add_argument("--no-bf16", action="store_true",
                       help="Disable bf16 even on CUDA.")
    parser.add_argument("--compile", action="store_true",
                       help="torch.compile the model (CUDA, may take 1-2 min to warm up).")
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def lr_schedule(step: int, max_steps: int, warmup_steps: int,
                base_lr: float, min_ratio: float) -> float:
    if step <= warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def get_or_train_tokenizer(data_path: Path, checkpoint_dir: Path, vocab_size: int,
                           bpe_max_chars: int) -> BPETokenizer:
    path = checkpoint_dir / "tokenizer.json"
    if path.exists():
        print(f"loading tokenizer from {path}")
        return BPETokenizer.load(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"training BPE tokenizer (vocab={vocab_size}, on {bpe_max_chars / 1e6:.0f} MB)...")
    start = time.time()
    chunk_iter = iter_cleaned_chunks(data_path, bpe_max_chars)
    tok = BPETokenizer.train(chunk_iter, vocab_size=vocab_size)
    tok.save(path)
    print(f"tokenizer trained in {format_duration(time.time() - start)}, vocab={tok.vocab_size}")
    return tok


def get_or_encode_ids(data_path: Path, tokenizer: BPETokenizer, checkpoint_dir: Path,
                     max_chars: int, source_signature: str) -> np.memmap:
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


def evaluate(model: MiniGPT, dataset: TokenDataset, batch_size: int, ctx: int,
             batches: int, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = dataset.next_batch(batch_size, ctx)
            x_t = torch.from_numpy(x).to(device)
            y_t = torch.from_numpy(y).to(device)
            loss = model.loss(x_t, y_t)
            total += loss.item()
            n += 1
    model.train()
    return total / max(1, n)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    use_bf16 = (device.type == "cuda" and not args.no_bf16) or args.bf16
    print(f"device={device} bf16={use_bf16}")

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
        model, cfg = load_checkpoint(args.checkpoint, device)
    else:
        print("starting new model")
        cfg = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=args.context_length,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            feed_forward_dim=args.feed_forward_dim,
            tie_embeddings=args.tie_embeddings,
        )
        model = MiniGPT(cfg)
        init_weights(model, cfg)
        model = model.to(device)

    if args.compile and device.type == "cuda":
        print("torch.compile() — first step will be slow (graph compilation)")
        model = torch.compile(model)

    n_params = param_count(model)
    print(f"vocab={tokenizer.vocab_size} params={n_params:,} "
          f"ctx={cfg.context_length} layers={cfg.num_layers} dim={cfg.embedding_dim}")

    train_ids, val_ids = split_train_val_ids(ids, args.val_fraction)
    train_set = TokenDataset.from_ids(train_ids, args.seed)
    val_set = TokenDataset.from_ids(val_ids, args.seed + 1)
    print(f"train tokens={len(train_ids):,} val tokens={len(val_ids):,}")

    # LLM-standard AdamW: betas (0.9, 0.95); weight decay only on matrices,
    # not on norms/embeddings/biases (GPT-3/LLaMA/DeepSeek practice).
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )

    # Resume optimizer state + step for exact continuation.
    start_step = 0
    best_val = float("inf")
    opt_path = args.checkpoint / "optimizer.pt"
    state_path = args.checkpoint / "train_state.json"
    if state_path.exists():
        import json as _json
        meta = _json.loads(state_path.read_text(encoding="utf-8"))
        start_step = int(meta.get("step", 0))
        best_val = float(meta.get("best_val", float("inf")))
        if opt_path.exists():
            try:
                optimizer.load_state_dict(torch.load(opt_path, map_location=device))
                print(f"restored optimizer state at step {start_step}")
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] optimizer state mismatch ({exc}); fresh optimizer")
    if start_step >= args.max_steps:
        print(f"checkpoint already at step {start_step} >= max-steps {args.max_steps}")
        return

    def save_train_state(step: int) -> None:
        import json as _json
        torch.save(optimizer.state_dict(), opt_path)
        state_path.write_text(_json.dumps({"step": step, "best_val": best_val}),
                              encoding="utf-8")

    # Sanity probe
    x0, y0 = train_set.next_batch(args.batch_size, cfg.context_length)
    with torch.no_grad():
        loss0 = model.loss(torch.from_numpy(x0).to(device), torch.from_numpy(y0).to(device))
    print(f"init forward loss={loss0.item():.4f} (random baseline ~{math.log(tokenizer.vocab_size):.2f})")

    started = time.time()
    last_log = started
    last_log_step = start_step
    recent_loss = 0.0
    recent_count = 0

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    amp_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if use_bf16 else _NullCtx()

    for step in range(start_step + 1, args.max_steps + 1):
        x_np, y_np = train_set.next_batch(args.batch_size, cfg.context_length)
        x = torch.from_numpy(x_np).to(device, non_blocking=True)
        y = torch.from_numpy(y_np).to(device, non_blocking=True)

        lr = lr_schedule(step, args.max_steps, args.warmup_steps, args.lr, args.min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with amp_ctx:
            loss = model.loss(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_v = loss.item()
        recent_loss += loss_v
        recent_count += 1

        if step == start_step + 1 or step % args.log_every == 0:
            now = time.time()
            avg = recent_loss / max(1, recent_count)
            ms_per_step = (now - last_log) * 1000.0 / max(1, step - last_log_step)
            eta = (now - started) / max(1, step - start_step) * (args.max_steps - step)
            print(
                f"step {step}/{args.max_steps} ({100.0 * step / args.max_steps:.1f}%) "
                f"loss {loss_v:.4f} (avg {avg:.4f}) gnorm {float(grad_norm):.2f} lr {lr:.2e} "
                f"{ms_per_step:.0f} ms/step ETA {format_duration(eta)}"
            )
            last_log = now
            last_log_step = step
            recent_loss = 0.0
            recent_count = 0

        if args.val_every and step % args.val_every == 0:
            val_loss = evaluate(model, val_set, args.batch_size, cfg.context_length,
                               args.val_batches, device)
            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(args.checkpoint, model, cfg,
                                weights_name="weights_best.safetensors")
                marker = " (best → weights_best.safetensors)"
            print(f"[val] step {step} loss {val_loss:.4f}{marker}")

        if args.checkpoint_every and step % args.checkpoint_every == 0:
            save_checkpoint(args.checkpoint, model, cfg)
            save_train_state(step)
            print(f"[checkpoint] step {step} saved to {args.checkpoint}")

    save_checkpoint(args.checkpoint, model, cfg)
    save_train_state(args.max_steps)
    print(f"saved final model to {args.checkpoint}")


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
