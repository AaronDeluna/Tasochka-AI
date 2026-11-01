"""Training script: BPE tokenizer + LLaMA-style mini transformer (MLX).

Steps:
    1. Load and clean a JSONL file, TXT file, or directory with TXT files.
    2. Train (or load) BPE tokenizer.
    3. Encode the corpus once; cache tokens to disk for fast resume.
    4. AdamW (betas 0.9/0.95) + global L2 grad clipping + warmup+cosine LR.

Efficiency (Apple Silicon, 16 GB):
    --bf16          mixed precision: вычисления в bf16 (~1.4x быстрее), но веса
                    и моменты Adam хранятся в fp32 (иначе мелкие апдейты
                    округляются в ноль и обучение встаёт на низком LR)
    mx.compile      train step компилируется в один Metal-граф (оба пути:
                    и обычный, и с накоплением градиента)
    --grad-accum N  эффективный батч = batch-size * N без роста памяти
    resume          состояние оптимизатора и номер шага сохраняются — обучение
                    продолжается ровно с того места, где остановилось
"""
from __future__ import annotations

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from .data import (
    TokenDataset,
    corpus_signature,
    encode_corpus_to_file,
    iter_cleaned_chunks,
    split_train_val_ids,
    token_storage_dtype,
)
from .model_mlx import (
    MiniGPT,
    ModelConfig,
    init_weights,
    load_checkpoint,
    param_count,
    save_checkpoint,
)
from .config import parse_args_with_config
from .tokenizer import BPETokenizer

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "datasets" / "en"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "tasochka"

ADAM_BETAS = (0.9, 0.95)  # LLM-стандарт (GPT-3, LLaMA, DeepSeek, Qwen)


class AdamWNoNormDecay(optim.AdamW):
    """AdamW that skips weight decay on 1-D parameters (RMSNorm gains).

    GPT-3/LLaMA/DeepSeek practice: decay matrices, but never the norm gains —
    they start at 1.0 and pulling them toward zero degrades normalisation.
    MLX's AdamW applies `weight_decay` to every parameter, so we override the
    per-parameter update; the PyTorch path does the same via param groups.
    """

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        if parameter.ndim < 2:
            return optim.Adam.apply_single(self, gradient, parameter, state)
        return super().apply_single(gradient, parameter, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1,
                       help="Gradient accumulation: effective batch = batch-size * grad-accum.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-dataset-chars", type=int, default=30_000_000)
    parser.add_argument("--vocab-size", type=int, default=32000,
                       help="Размер BPE-словаря. Для русского 8000 мало: 32k даёт "
                            "4.56 симв/токен против 3.69 -> на 19%% меньше токенов "
                            "на тот же текст. Для маленькой модели (dim<=896) бери 16000.")
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
    parser.add_argument("--bf16", action="store_true",
                       help="Mixed precision: считать в bf16 (~1.4x быстрее), веса и "
                            "моменты Adam держать в fp32. Рекомендуется всегда.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                       help="Cosine decay floor as fraction of --lr (final LR = lr * ratio).")
    parser.add_argument("--no-save-optimizer", action="store_true",
                       help="Не сохранять состояние оптимизатора (меньше диска, но resume неточный).")
    return parse_args_with_config(parser)


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
    """Encode corpus to a memmap-backed file. Re-uses cache if signature matches."""
    ids_path = checkpoint_dir / "corpus_ids.bin"
    meta_path = checkpoint_dir / "corpus_ids.meta"
    dtype = token_storage_dtype(tokenizer.vocab_size)
    # "v3" (v3 = new cleaning + proportional mixing) + dtype in the signature: an
    # older cache must not be reused, and an int32 cache must not be reinterpreted
    # as uint16, which would silently feed the model garbage token ids.
    meta_value = f"v3:{dtype.name}:{max_chars}:{tokenizer.vocab_size}:{source_signature}"
    if ids_path.exists() and meta_path.exists():
        meta = meta_path.read_text(encoding="utf-8").strip()
        if meta == meta_value:
            n_tokens = ids_path.stat().st_size // dtype.itemsize
            print(f"loading cached token ids from {ids_path} ({n_tokens / 1e6:.1f}M tokens)")
            return np.memmap(ids_path, dtype=dtype, mode="r", shape=(n_tokens,))
    print(f"encoding corpus (streaming, max {max_chars / 1e6:.0f} MB, {dtype.name})...")
    start = time.time()
    cleaned_chars, n_tokens = encode_corpus_to_file(
        data_path, tokenizer, max_chars, ids_path, dtype=dtype
    )
    meta_path.write_text(meta_value, encoding="utf-8")
    print(f"encoded in {format_duration(time.time() - start)}: "
          f"{cleaned_chars / 1e6:.1f}M cleaned chars → {n_tokens / 1e6:.1f}M tokens "
          f"(compression {cleaned_chars / max(1, n_tokens):.2f} chars/token, "
          f"{n_tokens * dtype.itemsize / 1e9:.2f} GB on disk)")
    return np.memmap(ids_path, dtype=dtype, mode="r", shape=(n_tokens,))


def save_train_state(directory: Path, optimizer, step: int, best_val: float,
                     save_optimizer: bool) -> None:
    if save_optimizer:
        flat = dict(tree_flatten(optimizer.state))
        mx.save_safetensors(str(directory / "optimizer.safetensors"), flat)
    (directory / "train_state.json").write_text(
        json.dumps({"step": step, "best_val": best_val}), encoding="utf-8"
    )


def _fill_state(node, saved: dict, prefix: str = "") -> None:
    """Copy saved leaves into an already-initialised optimizer state, in place.

    `optimizer.init()` builds the exact structure the optimizer will index into
    (flat param paths under mixed precision, a nested tree otherwise). Blindly
    `tree_unflatten`-ing the saved file would always produce the nested shape and
    silently mismatch the flat one, so we fill the existing structure instead.
    """
    items = node.items() if isinstance(node, dict) else enumerate(node)
    for key, value in list(items):
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, mx.array):
            if path not in saved:
                raise KeyError(f"optimizer state is missing {path}")
            node[key] = saved[path]
        elif isinstance(value, (dict, list)):
            _fill_state(value, saved, path)


def load_train_state(directory: Path, optimizer, model, init_params=None) -> tuple[int, float]:
    """Restore optimizer state + step. Returns (start_step, best_val).

    `init_params` is the tree the optimizer moments should be shaped/typed from —
    the fp32 master weights under mixed precision, so moments stay fp32.
    """
    state_path = directory / "train_state.json"
    if not state_path.exists():
        return 0, float("inf")
    meta = json.loads(state_path.read_text(encoding="utf-8"))
    step = int(meta.get("step", 0))
    best_val = float(meta.get("best_val", float("inf")))
    opt_path = directory / "optimizer.safetensors"
    if opt_path.exists():
        try:
            optimizer.init(model.trainable_parameters() if init_params is None
                           else init_params)
            _fill_state(optimizer.state, mx.load(str(opt_path)))
            mx.eval(optimizer.state)
            print(f"restored optimizer state at step {step}")
        except Exception as exc:  # noqa: BLE001 — arch mismatch → fresh optimizer
            print(f"[warn] optimizer state mismatch ({exc}); starting with fresh optimizer")
    return step, best_val


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
            num_kv_heads=args.num_kv_heads,
            feed_forward_dim=args.feed_forward_dim,
            tie_embeddings=args.tie_embeddings,
        )
        model = MiniGPT(cfg)
        init_weights(model, cfg, seed=args.seed)

    dtype = mx.bfloat16 if args.bf16 else mx.float32
    # Mixed precision, the way LLaMA/DeepSeek/Qwen do it: compute in bf16 but keep
    # an fp32 *master* copy of the weights for the optimizer update. Pure bf16
    # weights silently swallow small updates — bf16 has ~8 mantissa bits, so once
    # lr*update falls below ~0.4% of a weight it rounds straight back to the old
    # value and the model stops learning (worst at the end of the cosine decay).
    master = None
    if args.bf16:
        master = dict(tree_flatten(
            tree_map(lambda p: p.astype(mx.float32), model.trainable_parameters())
        ))
    model.set_dtype(dtype)
    mx.eval(model.parameters())

    print(f"vocab={tokenizer.vocab_size} params={param_count(model):,} "
          f"ctx={cfg.context_length} layers={cfg.num_layers} dim={cfg.embedding_dim} "
          f"heads={cfg.num_heads}/{cfg.num_kv_heads}kv tied={cfg.tie_embeddings} "
          f"dtype={'bf16' if args.bf16 else 'fp32'}")

    train_ids, val_ids = split_train_val_ids(ids, args.val_fraction)
    train_set = TokenDataset.from_ids(train_ids, args.seed)
    val_set = TokenDataset.from_ids(val_ids, args.seed + 1)
    print(f"train tokens={len(train_ids):,} val tokens={len(val_ids):,}")

    optimizer = AdamWNoNormDecay(
        learning_rate=args.lr,
        betas=list(ADAM_BETAS),
        weight_decay=args.weight_decay,
    )

    start_step, best_val = load_train_state(
        args.checkpoint, optimizer, model, init_params=master,
    )
    if start_step >= args.max_steps:
        print(f"checkpoint already at step {start_step} >= max-steps {args.max_steps}; "
              f"увеличь --max-steps чтобы продолжить")
        return

    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: m.loss(x, y))

    state = [model.state, optimizer.state] + ([master] if master is not None else [])

    def apply_update(grads):
        """Clip, then step the optimizer — in fp32 when master weights are on."""
        if master is None:
            grads, norm = optim.clip_grad_norm(grads, args.grad_clip)
            optimizer.update(model, grads)
            return norm
        # Work on flat {"blocks.0.attn.wq.weight": array} dicts. A flat dict is a
        # valid pytree and the keys always line up; nested trees contain
        # parameterless modules (RoPE) that flatten/unflatten silently drops,
        # which then makes tree_map trip over a missing key.
        flat_grads = {k: v.astype(mx.float32) for k, v in tree_flatten(grads)}
        flat_grads, norm = optim.clip_grad_norm(flat_grads, args.grad_clip)
        new_master = optimizer.apply_gradients(flat_grads, master)
        master.update(new_master)  # fp32 weights of record
        model.update(tree_unflatten(  # bf16 copy used for compute
            [(k, v.astype(dtype)) for k, v in new_master.items()]
        ))
        return norm

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x: mx.array, y: mx.array):
        loss, grads = loss_and_grad(model, x, y)
        return loss, apply_update(grads)

    @partial(mx.compile, inputs=state, outputs=state)
    def accum_step(batches):
        """Grad accumulation: average grads over micro-batches, then one update."""
        total_loss = mx.array(0.0)
        acc = None
        for bx, by in batches:
            loss, grads = loss_and_grad(model, bx, by)
            acc = grads if acc is None else tree_map(mx.add, acc, grads)
            total_loss = total_loss + loss
        scale = 1.0 / len(batches)
        acc = tree_map(lambda g: g * scale, acc)
        return total_loss * scale, apply_update(acc)

    # sanity probe
    x0, y0 = train_set.next_batch(args.batch_size, cfg.context_length)
    loss0 = model.loss(mx.array(x0), mx.array(y0))
    mx.eval(loss0)
    print(f"init forward loss={loss0.item():.4f} (random baseline ~{math.log(tokenizer.vocab_size):.2f})")
    if start_step:
        print(f"continuing from step {start_step}")

    tokens_per_step = args.batch_size * args.grad_accum * cfg.context_length
    started = time.time()
    last_log = started
    last_log_step = start_step
    recent_loss = 0.0
    recent_count = 0

    for step in range(start_step + 1, args.max_steps + 1):
        lr = lr_schedule(step, args.max_steps, args.warmup_steps, args.lr, args.min_lr_ratio)
        optimizer.learning_rate = lr

        if args.grad_accum > 1:
            batches = []
            for _ in range(args.grad_accum):
                bx, by = train_set.next_batch(args.batch_size, cfg.context_length)
                batches.append((mx.array(bx), mx.array(by)))
            loss, norm = accum_step(batches)
        else:
            x, y = train_set.next_batch(args.batch_size, cfg.context_length)
            loss, norm = train_step(mx.array(x), mx.array(y))
        mx.eval(loss, norm, state)
        loss_v = loss.item()
        recent_loss += loss_v
        recent_count += 1

        if step == start_step + 1 or step % args.log_every == 0:
            now = time.time()
            avg = recent_loss / max(1, recent_count)
            steps_done = max(1, step - last_log_step)
            ms_per_step = (now - last_log) * 1000.0 / steps_done
            tok_s = tokens_per_step * steps_done / max(1e-6, now - last_log)
            done = step - start_step
            eta = (now - started) / done * (args.max_steps - step)
            print(
                f"step {step}/{args.max_steps} ({100.0 * step / args.max_steps:.1f}%) "
                f"loss {loss_v:.4f} (avg {avg:.4f}) gnorm {float(norm):.2f} lr {lr:.2e} "
                f"{ms_per_step:.0f} ms/step {tok_s / 1000:.1f}k tok/s ETA {format_duration(eta)}"
            )
            last_log = now
            last_log_step = step
            recent_loss = 0.0
            recent_count = 0

        if args.val_every and step % args.val_every == 0:
            val_loss = evaluate(model, val_set, args.batch_size, cfg.context_length, args.val_batches)
            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(args.checkpoint, model, cfg,
                                weights_name="weights_best.safetensors",
                                params=master)
                marker = " (best → weights_best.safetensors)"
            print(f"[val] step {step} loss {val_loss:.4f}{marker}")

        if args.checkpoint_every and step % args.checkpoint_every == 0:
            save_checkpoint(args.checkpoint, model, cfg,
                            params=master)
            save_train_state(args.checkpoint, optimizer, step, best_val,
                             not args.no_save_optimizer)
            print(f"[checkpoint] step {step} saved to {args.checkpoint}")

    save_checkpoint(args.checkpoint, model, cfg,
                    params=master)
    save_train_state(args.checkpoint, optimizer, args.max_steps, best_val,
                     not args.no_save_optimizer)
    print(f"saved final model to {args.checkpoint}")


def evaluate(model: MiniGPT, dataset: TokenDataset, batch_size: int, ctx: int, batches: int) -> float:
    # Same windows every time — otherwise val loss is measured on different
    # data at each call and can't be compared across steps.
    dataset.reset_rng()
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
