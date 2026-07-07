#!/bin/bash
# Чат с этой версией модели. Пример: bash run.sh "Как тебя зовут?"
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$HOME/.tasochka-finetune-venv/bin/python"
BASE=$(awk '/^base_model:/{print $2}' "$DIR/config.yaml")
$PYTHON -m mlx_lm generate --model "$BASE" --adapter-path "$DIR/adapters" \
  --prompt "${1:-Как тебя зовут?}" --max-tokens "${MAX_TOKENS:-1536}" --temp "${TEMP:-0.7}"
