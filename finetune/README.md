# finetune/ — дообучение готовой модели в персону «Тасочка AI»

Два пайплайна:

1. **QLoRA (рекомендуемый)** — Gemma 3 4B в 4-bit + LoRA на все слои,
   обучение на реальных русских диалогах + persona. Модель после этого
   нормально разговаривает по-русски. См. [TRAINING_GUIDE.md](../TRAINING_GUIDE.md).
2. **Старый LoRA** — bf16 база из `base_model/`, только persona-датасет
   (учит имя, но не улучшает речь). Оставлен для совместимости.

## Структура

```
finetune/
├── build_persona.py        ← генератор persona-примеров (имя, «ты не Gemma» и т.д.)
├── build_chat_dataset.py   ← НОВОЕ: собирает разговорный датасет из
│                             russian-train-dataset + persona → data_chat/
├── qlora_config.yaml       ← НОВОЕ: все гиперпараметры QLoRA с комментариями
├── train_qlora.sh          ← НОВОЕ: запуск QLoRA (база скачается сама, ~2.5 GB)
├── chat.sh                 ← чат с адаптером (env MODEL/ADAPTERS для старого пайплайна)
├── fuse.sh                 ← слить адаптер в одну модель
├── data/                   ← persona-примеры (build_persona.py)
├── data_chat/              ← полный чат-датасет (build_chat_dataset.py), в .gitignore
├── adapters_qlora/         ← адаптеры QLoRA, в .gitignore
├── base_model/             ← старая bf16 Gemma (для старого пайплайна), в .gitignore
└── adapters/               ← старые адаптеры, в .gitignore
```

## Быстрый старт (QLoRA)

```bash
# 1. Собрать датасет (~5 минут)
./.venv/bin/python finetune/build_chat_dataset.py

# 2. Обучить (~2-4 часа на M4 16GB)
bash finetune/train_qlora.sh

# 3. Проверить
bash finetune/chat.sh "Как тебя зовут?"
bash finetune/chat.sh "Ты Gemma?"

# 4. (опционально) слить в одну модель
bash finetune/fuse.sh
```

Что менять при проблемах — таблица в [TRAINING_GUIDE.md](../TRAINING_GUIDE.md).

## Старый пайплайн (bf16 LoRA, только persona)

```bash
./.venv/bin/python finetune/build_persona.py
bash finetune/train_lora.sh
MODEL=finetune/base_model ADAPTERS=finetune/adapters bash finetune/chat.sh "Как тебя зовут?"
```
