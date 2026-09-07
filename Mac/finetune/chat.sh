#!/bin/bash
# Чат с базовой Gemma 3 + LoRA адаптерами (без слияния).
# Удобно для проверки до fuse.
set -e
cd "$(dirname "$0")/.."

PYTHON="$HOME/.tasochka-finetune-venv/bin/python"

PROMPT="${1:-Как тебя зовут?}"

$PYTHON -m mlx_lm generate \
  --model finetune/base_model \
  --adapter-path finetune/adapters \
  --prompt "$PROMPT" \
  --max-tokens 200 \
  --temp 0.7
