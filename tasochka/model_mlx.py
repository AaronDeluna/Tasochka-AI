"""MLX transformer language model — LLaMA-style mini version.

Architecture:
  - Token embedding (no learned positional embedding)
  - N blocks of:
      RMSNorm -> CausalAttention with RoPE (optionally GQA) -> residual ->
      RMSNorm -> SwiGLU FF -> residual
  - Final RMSNorm
  - LM head (no bias); optionally tied to the token embedding

Efficiency features (all backwards-compatible with old checkpoints):
  - GQA (grouped-query attention): fewer KV heads than Q heads, like
    Qwen/DeepSeek/LLaMA-3. Cuts KV projection params and attention memory.
    Enabled via `num_kv_heads` < `num_heads`; old configs default to MHA.
  - Tied embeddings: LM head reuses the token embedding matrix, saving
    vocab*dim parameters (standard in Qwen, Gemma, all small models).
  - Scaled init: normal(0, 0.02) with residual projections scaled down by
    1/sqrt(2*num_layers) (GPT-2/LLaMA practice) — stabler early training.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .checkpoint_io import atomic_write, atomic_write_text


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 256
    embedding_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    feed_forward_dim: int = 1536
    # 0 means "same as num_heads" (classic MHA) — keeps old checkpoints valid.
    num_kv_heads: int = 0
    tie_embeddings: bool = False
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.num_kv_heads in (0, None):
            self.num_kv_heads = self.num_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.num_heads


class KVCache:
    """Keys/values of one attention layer for incremental generation.

    The buffer is allocated once at `max_len` positions and filled in place, so
    every decode step reuses the same memory. Growing it by concatenation
    instead would allocate a buffer of a new size on every token, and MLX keeps
    each of those in its buffer cache — the leak that froze a 16 GB Mac.
    """

    def __init__(self, max_len: int):
        self.max_len = max_len
        self.offset = 0
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None

    def update(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        B, H, T, D = k.shape
        end = self.offset + T
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} > {self.max_len} positions")
        if self.keys is None:
            self.keys = mx.zeros((B, H, self.max_len, D), dtype=k.dtype)
            self.values = mx.zeros((B, H, self.max_len, D), dtype=v.dtype)
        self.keys[:, :, self.offset:end, :] = k
        self.values[:, :, self.offset:end, :] = v
        self.offset = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]


def make_cache(cfg: "ModelConfig", max_len: Optional[int] = None) -> List[KVCache]:
    return [KVCache(max_len or cfg.context_length) for _ in range(cfg.num_layers)]


class CausalAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.num_heads
        self.n_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        kv_dim = self.n_kv_heads * self.head_dim
        self.wq = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        self.wk = nn.Linear(cfg.embedding_dim, kv_dim, bias=False)
        self.wv = nn.Linear(cfg.embedding_dim, kv_dim, bias=False)
        self.wo = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        # traditional=True rotates adjacent pairs (x0,x1),(x2,x3)... — the same
        # layout as model_torch._apply_rope. With traditional=False MLX rotates
        # halves (x_i, x_{i+d/2}) instead, so a checkpoint trained on one backend
        # would see scrambled positions on the other (loss 2.76 vs 4.57).
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=cfg.rope_theta)

    def __call__(self, x: mx.array, cache: Optional[KVCache] = None) -> mx.array:
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update(k, v)
        if T == 1:
            mask = None  # a single new token may see every cached position
        elif offset == 0:
            # String mask lets Metal use the fused causal kernel (no T*T mask
            # alloc); GQA (n_kv_heads < n_heads) is handled natively by the kernel.
            mask = "causal"
        else:
            # Several new tokens on top of a filled cache: token i sees all
            # cached positions plus new tokens up to itself.
            q_pos = mx.arange(offset, offset + T)[:, None]
            k_pos = mx.arange(offset + T)[None, :]
            mask = mx.where(k_pos <= q_pos, 0.0, -mx.inf).astype(q.dtype)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.embedding_dim, cfg.feed_forward_dim, bias=False)
        self.w2 = nn.Linear(cfg.feed_forward_dim, cfg.embedding_dim, bias=False)
        self.w3 = nn.Linear(cfg.embedding_dim, cfg.feed_forward_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.embedding_dim)
        self.attn = CausalAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.embedding_dim)
        self.ff = SwiGLU(cfg)

    def __call__(self, x: mx.array, cache: Optional[KVCache] = None) -> mx.array:
        x = x + self.attn(self.norm1(x), cache)
        x = x + self.ff(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.blocks = [Block(cfg) for _ in range(cfg.num_layers)]
        self.norm_f = nn.RMSNorm(cfg.embedding_dim)
        if not cfg.tie_embeddings:
            self.head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)

    def __call__(self, idx: mx.array, cache: Optional[List[KVCache]] = None,
                 last_only: bool = False) -> mx.array:
        """Logits [B, T, vocab]. With `cache` (one KVCache per layer) only the new
        tokens in `idx` are processed; `last_only` returns logits for the last
        position only ([B, 1, vocab]) — generation needs nothing else, and the
        full [T, vocab] projection is the biggest tensor of a prefill."""
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cache[i] if cache is not None else None)
        if last_only:
            x = x[:, -1:, :]
        x = self.norm_f(x)
        if self.cfg.tie_embeddings:
            return self.tok_emb.as_linear(x)
        return self.head(x)

    def loss(self, idx: mx.array, targets: mx.array) -> mx.array:
        logits = self(idx)
        B, T, V = logits.shape
        # Cross-entropy in fp32 even when the model runs in bf16.
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V).astype(mx.float32),
            targets.reshape(B * T),
            reduction="mean",
        )


def init_weights(model: MiniGPT, cfg: ModelConfig, seed: int = 0) -> None:
    """GPT-2/LLaMA-style init: normal(0, 0.02); residual output projections
    (attn.wo, ff.w2) scaled down by 1/sqrt(2*num_layers)."""
    mx.random.seed(seed)
    std = 0.02
    resid_std = std / math.sqrt(2 * cfg.num_layers)

    def _init(name: str, module) -> None:
        if isinstance(module, nn.Linear):
            s = resid_std if name.endswith((".wo", ".w2")) else std
            module.weight = s * mx.random.normal(module.weight.shape)
        elif isinstance(module, nn.Embedding):
            module.weight = std * mx.random.normal(module.weight.shape)

    for name, module in model.named_modules():
        _init(name, module)
    mx.eval(model.parameters())


def _flatten(obj, prefix: str, out: dict) -> None:
    if isinstance(obj, mx.array):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            _flatten(v, key, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}.{i}" if prefix else str(i)
            _flatten(v, key, out)


def _unflatten(flat: dict) -> dict:
    root: dict = {}
    for key, value in flat.items():
        parts = key.split(".")
        node = root
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            next_part = parts[i + 1] if not last else None
            child_is_list = next_part is not None and next_part.isdigit()
            if part.isdigit():
                idx = int(part)
                while len(node) <= idx:
                    node.append(None)
                if last:
                    node[idx] = value
                else:
                    if node[idx] is None:
                        node[idx] = [] if child_is_list else {}
                    node = node[idx]
            else:
                if last:
                    node[part] = value
                else:
                    if part not in node or node[part] is None:
                        node[part] = [] if child_is_list else {}
                    node = node[part]
    return root


def save_checkpoint(directory: Path, model: MiniGPT, cfg: ModelConfig,
                    weights_name: str = "weights.safetensors",
                    params: dict | None = None) -> None:
    """Save weights + config. `params` overrides what gets written — used to
    store the fp32 master weights instead of the bf16 compute copy, so restarting
    a run does not quantise the model down a little every time."""
    directory.mkdir(parents=True, exist_ok=True)
    weights = dict(model.parameters()) if params is None else params
    flat: dict = {}
    _flatten(weights, "", flat)
    atomic_write(directory / weights_name, lambda path: mx.save_safetensors(path, flat))
    atomic_write_text(directory / "config.json",
                      json.dumps(asdict(cfg), ensure_ascii=False, indent=2))


def load_checkpoint(directory: Path) -> tuple[MiniGPT, ModelConfig]:
    raw_cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    # Ignore config keys this code version doesn't know (forward compatibility).
    known = set(ModelConfig.__dataclass_fields__)
    cfg = ModelConfig(**{k: v for k, v in raw_cfg.items() if k in known})
    model = MiniGPT(cfg)
    flat = mx.load(str(directory / "weights.safetensors"))
    nested = _unflatten(flat)
    model.update(nested)
    mx.eval(model.parameters())
    return model, cfg


def param_count(model: MiniGPT) -> int:
    total = 0
    flat: dict = {}
    _flatten(dict(model.parameters()), "", flat)
    for arr in flat.values():
        total += int(math.prod(arr.shape))
    return total
