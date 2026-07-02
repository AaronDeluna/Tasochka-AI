# Tasochka AI

Русскоязычный ассистент «Тасочка». Два независимых проекта в одном репозитории:

```
Tasochka AI/
├── finetune/                ← ПУТЬ 1 (рекомендуемый): QLoRA-дообучение Gemma 3 4B.
│                              Модель после этого нормально общается по-русски.
├── from-scratch/            ← ПУТЬ 2 (учебный): свой мини-GPT с нуля на MLX/PyTorch.
│   ├── tasochka/              Код модели и тренера.
│   └── checkpoints/           Чекпойнты (не в git).
├── russian-train-dataset/   ← Общие данные для обоих путей (не в git, ~8.5 GB).
├── download_datasets.py     ← Скачивает датасеты с HuggingFace в russian-train-dataset/.
├── TRAINING_GUIDE.md        ← ГЛАВНЫЙ ДОК: готовые команды для Mac mini M4 16GB.
└── .venv/                   ← Общее окружение (./setup.sh).
```

**Все команды запуска — в [TRAINING_GUIDE.md](TRAINING_GUIDE.md).** Коротко:

## Путь 1: QLoRA — ассистент, который умеет разговаривать (~2-4 часа)

```bash
cd "/Users/ivanmilovanov/Desktop/Tasochka AI" && ./.venv/bin/python finetune/build_chat_dataset.py
cd "/Users/ivanmilovanov/Desktop/Tasochka AI" && bash finetune/train_qlora.sh
bash finetune/chat.sh "Привет! Как тебя зовут?"
```

Детали: [finetune/README.md](finetune/README.md).

## Путь 2: свой GPT с нуля (учебный)

BPE токенизация · RMSNorm · RoPE · SwiGLU · GQA · tied embeddings · bf16 · mx.compile.

```bash
cd "/Users/ivanmilovanov/Desktop/Tasochka AI/from-scratch" && ../.venv/bin/python -m tasochka.train --bf16 --tie-embeddings --embedding-dim 768 --num-layers 12 --num-heads 12 --num-kv-heads 4 --feed-forward-dim 2048 --context-length 512 --batch-size 8 --grad-accum 2 --max-steps 30000 --lr 3e-4 --warmup-steps 1000 --max-dataset-chars 2000000000
```

Общение и веб-интерфейс:

```bash
cd "/Users/ivanmilovanov/Desktop/Tasochka AI/from-scratch" && ../.venv/bin/python -m tasochka.generate --chat
cd "/Users/ivanmilovanov/Desktop/Tasochka AI/from-scratch" && ../.venv/bin/python -m tasochka.server
```

(веб-чат: открой `from-scratch/index.html`, сервер говорит по Ollama-протоколу на 127.0.0.1:11434)

## Первичная настройка (один раз)

```bash
cd "/Users/ivanmilovanov/Desktop/Tasochka AI" && ./setup.sh
```

Датасеты (если ещё не скачаны): `./.venv/bin/python download_datasets.py`

## Ожидания по качеству

- **QLoRA (Путь 1)**: уровень нормального ассистента — связная русская речь,
  держит контекст, знает что она «Тасочка». Это готовая 4B-модель, дообученная
  под персону на отфильтрованных русских диалогах.
- **С нуля (Путь 2)**: 80-200M параметров на домашнем железе = «пишет осмысленные
  предложения, держит тему 2-3 реплики». Это учебный проект про то, как LLM
  устроены изнутри, а не замена ChatGPT.
