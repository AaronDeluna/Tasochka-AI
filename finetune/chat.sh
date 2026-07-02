#!/bin/bash
# Чат с моделью + LoRA адаптерами (без слияния).
# По умолчанию — новый QLoRA-пайплайн. Старый вариант:
#   MODEL=finetune/base_model ADAPTERS=finetune/adapters bash finetune/chat.sh "..."
set -e
cd "$(dirname "$0")/.."

PYTHON="$HOME/.tasochka-finetune-venv/bin/python"
MODEL="${MODEL:-mlx-community/gemma-3-4b-it-qat-4bit}"
ADAPTERS="${ADAPTERS:-finetune/adapters_qlora}"

PROMPT="${1:-Как тебя зовут?}"

$PYTHON -m mlx_lm generate \
  --model "$MODEL" \
  --adapter-path "$ADAPTERS" \
  --prompt "$PROMPT" \
  --max-tokens 300 \
  --temp 0.7
