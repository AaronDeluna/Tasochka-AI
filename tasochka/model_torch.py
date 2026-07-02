"""PyTorch port of the LLaMA-style mini transformer.

Same architecture as model_mlx.py (RMSNorm, RoPE, SwiGLU, causal attention)
and the same checkpoint format (safetensors), so weights trained on Mac via
MLX can be loaded here and vice versa.

Device selection:
    PyTorch chooses CUDA > MPS > CPU automatically via `pick_device()`.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 256
    embedding_dim: int = 896
    num_layers: int = 20
    num_heads: int = 14
    feed_forward_dim: int = 2400
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


def pick_device(prefer: Optional[str] = None) -> torch.device:
    """Choose CUDA > MPS > CPU unless `prefer` is set."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _precompute_rope(head_dim: int, max_seq: int, base: float = 10000.0,
                    device: Optional[torch.device] = None,
                    dtype: Optional[torch.dtype] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin) tables of shape [max_seq, head_dim] for RoPE."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [max_seq, head_dim/2]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    # interleave to [max_seq, head_dim]
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    if device is not None:
        cos = cos.to(device)
        sin = sin.to(device)
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape [B, H, T, D]. cos/sin are [T, D]."""
    # rotate pairs: (x_0, x_1) -> (-x_1, x_0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * cos + rotated * sin


class RMSNorm(nn.Module):
    """RMSNorm: like LayerNorm but without subtracting the mean."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class CausalAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.num_heads
        self.n_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        self.wq = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        self.wk = nn.Linear(cfg.embedding_dim, kv_dim, bias=False)
        self.wv = nn.Linear(cfg.embedding_dim, kv_dim, bias=False)
        self.wo = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos[:T], sin[:T])
        k = _apply_rope(k, cos[:T], sin[:T])
        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # scaled_dot_product_attention picks Flash Attention when available
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.embedding_dim, cfg.feed_forward_dim, bias=False)
        self.w2 = nn.Linear(cfg.feed_forward_dim, cfg.embedding_dim, bias=False)
        self.w3 = nn.Linear(cfg.embedding_dim, cfg.feed_forward_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.embedding_dim)
        self.attn = CausalAttention(cfg)
        self.norm2 = RMSNorm(cfg.embedding_dim)
        self.ff = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ff(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.norm_f = RMSNorm(cfg.embedding_dim)
        if not cfg.tie_embeddings:
            self.head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)
        # RoPE cache lazily initialised on first forward (so we know device/dtype)
        self._rope_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._rope_max_seq: int = 0

    def _ensure_rope(self, seq: int, device: torch.device, dtype: torch.dtype):
        if self._rope_cache is None or self._rope_max_seq < seq:
            cos, sin = _precompute_rope(self.cfg.head_dim, max(seq, self.cfg.context_length),
                                       base=self.cfg.rope_theta,
                                       device=device, dtype=dtype)
            self._rope_cache = (cos, sin)
            self._rope_max_seq = cos.shape[0]

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx)
        self._ensure_rope(T, x.device, x.dtype)
        cos, sin = self._rope_cache
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        if self.cfg.tie_embeddings:
            return F.linear(x, self.tok_emb.weight)
        return self.head(x)

    def loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.forward(idx)
        B, T, V = logits.shape
        return F.cross_entropy(logits.view(B * T, V), targets.view(B * T).long())


def save_checkpoint(directory: Path, model: MiniGPT, cfg: ModelConfig,
                    weights_name: str = "weights.safetensors") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # Convert state_dict to contiguous CPU tensors (safetensors requires contiguous)
    state = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, str(directory / weights_name))
    (directory / "config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def init_weights(model: MiniGPT, cfg: ModelConfig) -> None:
    """GPT-2/LLaMA-style init: normal(0, 0.02); residual output projections
    (attn.wo, ff.w2) scaled down by 1/sqrt(2*num_layers)."""
    std = 0.02
    resid_std = std / math.sqrt(2 * cfg.num_layers)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            s = resid_std if name.endswith((".wo", ".w2")) else std
            nn.init.normal_(module.weight, mean=0.0, std=s)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)


def load_checkpoint(directory: Path, device: Optional[torch.device] = None) -> tuple[MiniGPT, ModelConfig]:
    raw_cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    known = set(ModelConfig.__dataclass_fields__)
    cfg = ModelConfig(**{k: v for k, v in raw_cfg.items() if k in known})
    model = MiniGPT(cfg)
    state = load_file(str(directory / "weights.safetensors"))
    # Allow loading MLX-saved weights: MLX stores parameters under nested-key
    # naming like "blocks.0.attn.wq.weight" which matches PyTorch state_dict
    # already; tok_emb / head naming also matches. We just call load_state_dict.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_checkpoint] missing={missing[:3]}{'...' if len(missing)>3 else ''}, "
              f"unexpected={unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
    if device is not None:
        model = model.to(device)
    return model, cfg


def param_count(model: MiniGPT) -> int:
    return sum(p.numel() for p in model.parameters())
