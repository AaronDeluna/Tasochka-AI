#!/bin/bash
# Прогон 100 вопросов по 10 темам через base + LoRA адаптер.
# Результаты в finetune/eval_runs/eval_<timestamp>.md
set -e
cd "$(dirname "$0")/.."
"$HOME/.tasochka-finetune-venv/bin/python" finetune/run_eval.py
