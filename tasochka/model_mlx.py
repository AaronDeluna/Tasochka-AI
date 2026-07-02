"""MLX transformer language model — LLaMA-style mini version.

Architecture:
  - Token embedding (no learned positional embedding)
  - N blocks of:
      RMSNorm -> CausalAttention with RoPE -> residual ->
      RMSNorm -> SwiGLU FF -> residual
  - Final RMSNorm
  - LM head (no bias)
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 256
    embedding_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    feed_forward_dim: int = 1536

    def __post_init__(self):
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.num_heads


class CausalAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        self.wk = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        self.wv = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        self.wo = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(q)
        k = self.rope(k)
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

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.blocks = [Block(cfg) for _ in range(cfg.num_layers)]
        self.norm_f = nn.RMSNorm(cfg.embedding_dim)
        self.head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)

    @staticmethod
    def causal_mask(T: int) -> mx.array:
        return mx.triu(mx.full((T, T), -1e9), k=1)

    def __call__(self, idx: mx.array) -> mx.array:
        x = self.tok_emb(idx)
        mask = self.causal_mask(idx.shape[1])
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm_f(x)
        return self.head(x)

    def loss(self, idx: mx.array, targets: mx.array) -> mx.array:
        logits = self(idx)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
        )


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


def save_checkpoint(directory: Path, model: MiniGPT, cfg: ModelConfig) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    weights = dict(model.parameters())
    flat: dict = {}
    _flatten(weights, "", flat)
    mx.save_safetensors(str(directory / "weights.safetensors"), flat)
    (directory / "config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_checkpoint(directory: Path) -> tuple[MiniGPT, ModelConfig]:
    cfg = ModelConfig(**json.loads((directory / "config.json").read_text(encoding="utf-8")))
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
