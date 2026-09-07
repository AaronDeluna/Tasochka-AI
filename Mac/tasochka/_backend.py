"""Backend resolver: figures out whether to use MLX or PyTorch.

Order of preference:
  1. Explicit --backend flag (mlx|torch) from argv
  2. TASOCHKA_BACKEND env var
  3. Auto-detect: prefer MLX on Apple Silicon, fall back to PyTorch
"""
from __future__ import annotations

import os
import sys
from typing import Tuple


def _strip_backend_arg(argv: list[str]) -> Tuple[str, list[str]]:
    """Return (backend|"", remaining_argv) after consuming --backend value."""
    out: list[str] = []
    backend = ""
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--backend":
            if i + 1 < len(argv):
                backend = argv[i + 1]
                i += 2
                continue
        elif a.startswith("--backend="):
            backend = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    return backend, out


def _auto_detect() -> str:
    try:
        import mlx.core  # noqa: F401
        return "mlx"
    except ImportError:
        pass
    try:
        import torch  # noqa: F401
        return "torch"
    except ImportError:
        pass
    raise SystemExit(
        "Neither MLX nor PyTorch found. Install one:\n"
        "  pip install mlx                  # for Mac (Apple Silicon)\n"
        "  pip install torch                # for NVIDIA server or Mac CPU/MPS"
    )


def resolve() -> str:
    """Pop --backend from sys.argv and return chosen backend name.

    Modifies sys.argv in place so downstream argparse doesn't see --backend.
    """
    explicit, remaining = _strip_backend_arg(sys.argv)
    sys.argv = remaining

    if explicit:
        backend = explicit
    elif os.environ.get("TASOCHKA_BACKEND"):
        backend = os.environ["TASOCHKA_BACKEND"]
    else:
        backend = _auto_detect()

    if backend not in ("mlx", "torch"):
        raise SystemExit(f"Unknown backend: {backend!r}. Use 'mlx' or 'torch'.")
    return backend
