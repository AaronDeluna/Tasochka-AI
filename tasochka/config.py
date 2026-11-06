"""JSON-пресеты для тренеров: `--config configs/mac-smoke.json`.

Ключи пресета — это имена CLI-флагов через подчёркивание (`max_steps`,
`embedding_dim`, ...). Пресет задаёт значения по умолчанию, а флаги в командной
строке их перебивают:

    python -m tasochka.train --config configs/mac-1h.json --max-steps 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

# Ключи, которые понимает только один из бэкендов (torch: --device, --compile...).
# Пресет может их содержать — на другом бэкенде они молча игнорируются.
_PORTABLE_EXTRA = {"backend", "device", "compile", "grad_checkpoint", "no_bf16", "no_save_optimizer"}


def parse_args_with_config(parser: argparse.ArgumentParser,
                           argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser.add_argument("--config", type=Path, default=None,
                        help="JSON-пресет с параметрами (см. configs/). Флаги CLI его перебивают.")
    pre, _ = parser.parse_known_args(argv)
    if pre.config is not None:
        preset = json.loads(pre.config.read_text(encoding="utf-8"))
        preset = {k: v for k, v in preset.items() if not k.startswith("_")}  # "_comment"
        known = {a.dest for a in parser._actions}
        unknown = set(preset) - known - _PORTABLE_EXTRA
        if unknown:
            raise SystemExit(f"{pre.config}: неизвестные ключи {sorted(unknown)}")
        for key in ("data", "checkpoint"):
            if key in preset:
                preset[key] = Path(preset[key])
        parser.set_defaults(**{k: v for k, v in preset.items() if k in known})
        print(f"config: {pre.config}", file=sys.stderr)
    return parser.parse_args(argv)
