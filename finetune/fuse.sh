#!/bin/bash
# Слить LoRA-адаптеры в одну модель.
# По умолчанию — новый QLoRA-пайплайн. Старый вариант:
#   MODEL=finetune/base_model ADAPTERS=finetune/adapters OUT=finetune/tasochka_gemma bash finetune/fuse.sh
set -e
cd "$(dirname "$0")/.."
PYTHON="$HOME/.tasochka-finetune-venv/bin/python"
MODEL="${MODEL:-mlx-community/gemma-3-4b-it-qat-4bit}"
ADAPTERS="${ADAPTERS:-finetune/adapters_persona}"
OUT="${OUT:-finetune/tasochka_gemma}"

$PYTHON -m mlx_lm fuse --model "$MODEL" --adapter-path "$ADAPTERS" --save-path "$OUT"
echo "Готово. Слитая модель в $OUT"
