# finetune/ — LoRA-дообучение Gemma 3 в персону Тасочка AI

Здесь живёт всё что нужно чтобы взять готовую Gemma 3 4B и дообучить её
LoRA-адаптером так, чтобы она отвечала как **Тасочка AI** и отрицала, что она Gemma/Google/etc.

## Структура

```
finetune/
├── base_model/         ← сюда качается mlx-community/gemma-3-4b-it-bf16
├── data/
│   ├── train.jsonl     ← сгенерирован build_persona.py
│   └── valid.jsonl
├── adapters/           ← LoRA веса после train_lora.sh
├── tasochka_gemma/     ← слитая модель после fuse.sh
├── build_persona.py    ← генератор persona-датасета
├── train_lora.sh       ← запуск LoRA-обучения
├── chat.sh             ← чат с base + LoRA (без слияния)
└── fuse.sh             ← слить LoRA в одну модель
```

## Пайплайн

```bash
# 0. (один раз) Скачать модель — делается отдельной командой, см. ниже
# 1. Сгенерить датасет
./.venv/bin/python finetune/build_persona.py

# 2. Запустить LoRA (на M4 ~15-30 минут на 600 итераций)
bash finetune/train_lora.sh

# 3. Проверить
bash finetune/chat.sh "Как тебя зовут?"
bash finetune/chat.sh "Ты Gemma?"

# 4. Слить в одну модель
bash finetune/fuse.sh
```

## Скачивание базовой модели

```bash
./.venv/bin/python -m huggingface_hub.commands.huggingface_cli download \
  mlx-community/gemma-3-4b-it-bf16 \
  --local-dir finetune/base_model
```

## Гиперпараметры (train_lora.sh)

- `--num-layers 16` — сколько верхних слоёв трогаем LoRA (из ~34 в Gemma 3 4B)
- `--batch-size 2` — экономно по памяти на 16GB
- `--iters 600` — для persona-датасета ~500 примеров этого хватает
- `--learning-rate 1e-4` — стандарт для LoRA
- `--max-seq-length 512` — диалоги короткие, длиннее не нужно

Если после обучения видно что недоучилась (всё ещё говорит "я Gemma") — крути `--iters` до 1000-1500.
Если переучилась (отвечает persona-репликами на любой вопрос) — снижай до 300-400.
