"""Crash-safe checkpoint files, shared by the MLX and PyTorch trainers.

A run on a rented server can die at any moment (OOM killer, reboot, closed
SSH session). Three guarantees:

  * `atomic_write` — a file is written next to its target and renamed over it,
    so a crash mid-write leaves the previous checkpoint intact instead of a
    truncated one that can't be loaded.
  * `save_snapshot` — every N steps the weights are also copied into
    snapshots/step_NNNNNN/, so a run that diverges late can be rolled back.
  * `interrupt_on_termination` — SIGTERM/SIGHUP become KeyboardInterrupt, so
    the trainer's Ctrl+C handler saves the current step before exiting.
"""
from __future__ import annotations

import os
import shutil
import signal
from pathlib import Path
from typing import Callable


def atomic_write(path: Path, write: Callable[[str], None]) -> None:
    """Call `write(tmp_path)`, then atomically rename the result to `path`."""
    # Prefix, not suffix: keeps the extension (mx.save_safetensors appends
    # ".safetensors" to any path that doesn't already end with it).
    tmp = path.with_name(".tmp-" + path.name)
    write(str(tmp))
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write(path, lambda p: Path(p).write_text(text, encoding="utf-8"))


def save_snapshot(checkpoint_dir: Path, step: int) -> Path:
    """Copy weights + config + tokenizer to snapshots/step_NNNNNN/.

    The snapshot is a complete checkpoint directory: it can be passed straight
    to `tasochka.generate --checkpoint`, or copied back to resume from it.
    """
    dst = checkpoint_dir / "snapshots" / f"step_{step:06d}"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("weights.safetensors", "config.json", "tokenizer.json"):
        src = checkpoint_dir / name
        if src.exists():
            atomic_write(dst / name, lambda p, s=src: shutil.copyfile(s, p))
    return dst


def interrupt_on_termination() -> None:
    def handler(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, handler)
