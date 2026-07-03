#!/bin/bash
# Дообучение: персона Тасочки + Java-навык из дистилляции GPT.
# Собирает датасет (персона + балласт + дистиллированные Java Q&A) и учит LoRA.
#
# Перед запуском один раз собери Java-датасет через GPT:
#   ./.venv/bin/python finetune/distill_questions.py
#   export OPENAI_API_KEY="sk-..."
#   ./.venv/bin/python finetune/distill_gpt.py
set -e
cd "$(dirname "$0")/.."
PYTHON="$HOME/.tasochka-finetune-venv/bin/python"

if [ ! -f finetune/distill/java_qa.jsonl ]; then
  echo "Нет finetune/distill/java_qa.jsonl — сначала прогони дистилляцию:"
  echo "  ./.venv/bin/python finetune/distill_questions.py"
  echo "  export OPENAI_API_KEY=\"sk-...\" && ./.venv/bin/python finetune/distill_gpt.py"
  exit 1
fi

echo "Собираю датасет (персона + балласт + Java из дистилляции) ..."
./.venv/bin/python finetune/build_chat_dataset.py --preset persona \
  --extra-jsonl finetune/distill/java_qa.jsonl --out data_persona_code

# Больше данных → чуть больше итераций, чем в чистой персоне.
$PYTHON -m mlx_lm lora -c finetune/qlora_persona_config.yaml \
  --data finetune/data_persona_code \
  --adapter-path finetune/adapters_persona_code \
  --iters 1200 "$@"
echo "Готово. Адаптеры в finetune/adapters_persona_code"
echo "Проверка: ADAPTERS=finetune/adapters_persona_code bash finetune/chat.sh \"Напиши на Java метод проверки палиндрома\""
