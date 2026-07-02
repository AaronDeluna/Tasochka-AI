"""Training entry point. Routes to MLX or PyTorch implementation.

Usage (same CLI on both):
    # Auto-detect (MLX on Mac, PyTorch on server):
    python -m tasochka.train --max-steps 150000 --batch-size 5 ...

    # Explicit backend:
    python -m tasochka.train --backend mlx --max-steps 150000 ...
    python -m tasochka.train --backend torch --max-steps 150000 ...

    # Or via env var:
    TASOCHKA_BACKEND=torch python -m tasochka.train ...
"""
from ._backend import resolve


def main() -> None:
    backend = resolve()
    if backend == "mlx":
        from . import train_mlx
        train_mlx.main()
    else:
        from . import train_torch
        train_torch.main()


if __name__ == "__main__":
    main()
