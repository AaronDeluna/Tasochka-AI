#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PYTHON="$HOME/.tasochka-finetune-venv/bin/python"
$PYTHON -m mlx_lm fuse --model finetune/base_model --adapter-path finetune/adapters --save-path finetune/tasochka_gemma
echo "Готово. Слитая модель в finetune/tasochka_gemma"
