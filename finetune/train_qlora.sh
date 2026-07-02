#!/bin/bash
# QLoRA-дообучение Gemma 3 4B (4-bit) на русском чат-датасете + persona.
# Модель (~2.5 GB) скачается автоматически с HuggingFace при первом запуске.
#
# Перед первым запуском собери датасет:
#   ./.venv/bin/python finetune/build_chat_dataset.py
#
# Аргументы пробрасываются в mlx_lm, например:
#   bash finetune/train_qlora.sh --iters 3000
set -e
cd "$(dirname "$0")/.."
PYTHON="$HOME/.tasochka-finetune-venv/bin/python"

if [ ! -f finetune/data_chat/train.jsonl ]; then
  echo "Датасет не найден — собираю finetune/data_chat/ ..."
  ./.venv/bin/python finetune/build_chat_dataset.py
fi

$PYTHON -m mlx_lm lora -c finetune/qlora_config.yaml "$@"
echo "Готово. Адаптеры в finetune/adapters_qlora"
