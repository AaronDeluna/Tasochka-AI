#!/bin/bash
# Дообучение ТОЛЬКО персоны Тасочки поверх Gemma 3 4B (щадящий режим).
# Модель уже умеет русский и QA — учим её только личности, не ломая речь.
#
# Аргументы пробрасываются в mlx_lm, например:
#   bash finetune/train_persona.sh --iters 800
set -e
cd "$(dirname "$0")/.."
PYTHON="$HOME/.tasochka-finetune-venv/bin/python"

# Пересобираем persona-датасет всегда — он маленький и быстрый (~1 минута).
echo "Собираю persona-датасет (finetune/data_persona/) ..."
./.venv/bin/python finetune/build_chat_dataset.py --preset persona

$PYTHON -m mlx_lm lora -c finetune/qlora_persona_config.yaml "$@"
echo "Готово. Адаптеры в finetune/adapters_persona"
echo "Проверка: ADAPTERS=finetune/adapters_persona bash finetune/chat.sh \"Как тебя зовут?\""
