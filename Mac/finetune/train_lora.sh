#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PYTHON="$HOME/.tasochka-finetune-venv/bin/python"
$PYTHON -m mlx_lm lora --model finetune/base_model --train --data finetune/data --adapter-path finetune/adapters --batch-size 2 --num-layers 16 --iters 600 --learning-rate 1e-4 --steps-per-report 25 --steps-per-eval 100 --val-batches 20 --save-every 200 --max-seq-length 512 --seed 42
echo "Готово. Адаптеры в finetune/adapters"
