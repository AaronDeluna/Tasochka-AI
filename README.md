# Tasochka AI

Маленький GPT в стиле LLaMA, обученный с нуля на Apple GPU через MLX.

**Архитектура:** BPE токенизация · RMSNorm · RoPE · SwiGLU · 6 transformer-блоков · ~25M параметров.

## Быстрый старт

```bash
cd "/Users/ivanmilovanov/Desktop/Tasochka AI"
./setup.sh                # один раз
source .venv/bin/activate
```

### Тренировка

```bash
python -m tasochka.train --max-steps 30000 --batch-size 16
```

Первый запуск делает три вещи последовательно:
1. Читает и чистит `.txt` файлы из `russian-train-dataset/`
2. Тренирует BPE токенизатор (vocab=8000)
3. Кодирует корпус в токены (кэширует на диск)
4. Тренирует трансформер

При повторном запуске токенизатор и кэш токенов подтянутся, продолжится только train.

Дефолты:
```
--max-steps 10000          # шагов
--batch-size 16            # на M4 16GB
--lr 3e-4                  # с warmup на 100 шагов
--max-dataset-chars 30M
--vocab-size 8000          # BPE
--context-length 256       # тут BPE токенов = ~600-1000 слов
--embedding-dim 512
--num-layers 6
--num-heads 8
--feed-forward-dim 1536    # SwiGLU
--val-every 200
--checkpoint-every 500
--grad-clip 1.0
--weight-decay 0.01
--warmup-steps 100
```

По умолчанию обучение берет датасеты из:
```bash
russian-train-dataset/*.txt
```

Можно явно указать другой файл или папку:
```bash
python -m tasochka.train --data russian-train-dataset --max-steps 30000 --batch-size 16
```

Полный train (что я рекомендую):
```bash
python -m tasochka.train --max-steps 30000 --batch-size 16
```
Ожидаемое время на M4: 15–30 минут.

### Общение

```bash
python -m tasochka.generate --chat
```

Одиночный запрос:
```bash
python -m tasochka.generate "- Привет"
```

Параметры:
```bash
python -m tasochka.generate --chat --temperature 0.7 --top-k 40 --max-new 200
```

## Веб-интерфейс (опционально)

```bash
pip install -r requirements-server.txt
python -m tasochka.server
open index.html
```

Сервер слушает `127.0.0.1:11434`, говорит по Ollama-протоколу (NDJSON стрим). Игнорируется при консольной работе.

## Файлы чекпойнта

`checkpoints/tasochka/`:
- `tokenizer.json` — BPE
- `config.json` — конфиг модели
- `weights.safetensors` — веса
- `corpus_ids.npy` — закэшированные токены корпуса (пересоздаётся если изменилось)

Чтобы переучить с нуля — `rm -rf checkpoints/tasochka`.

## Архитектура файлов

```
tasochka/
├── tokenizer.py    # BPE на HuggingFace tokenizers
├── data.py         # TXT/JSONL стрим + cleaning
├── model.py        # MiniGPT: RMSNorm + RoPE + SwiGLU
├── train.py        # AdamW + grad clip + warmup + checkpoints
├── generate.py     # CLI + интерактивный чат
└── server.py       # опциональный HTTP API
```

## Ожидаемые цифры на M4 16GB

| Конфиг | Шаг | 10k шагов | val loss |
|---|---|---|---|
| default (25M, ctx=256, BPE) | 80–150 мс | ~15 мин | 2.5–3.0 |
| big (50M, ctx=512) | 200–400 мс | ~50 мин | 2.2–2.5 |

Char-level загнивал на val ~1.4 — это не сравнимо: разные единицы. Грубо: BPE val 2.7 ≈ char val 1.0 по качеству.

## Что улучшилось vs char-level

- **Связные слова всегда правильные** — BPE не может ошибиться в букве, выдаёт целое слово
- **Длинный контекст в словах** — 256 BPE-токенов это ~150-250 русских слов, против 25-30 в char-128
- **RoPE** даёт хорошее extrapolation за пределы контекста обучения
- **SwiGLU** учится лучше GELU при том же бюджете параметров
- **RMSNorm** стабильнее в FP16, чем LayerNorm

## Стеклянный потолок

25M параметров на ноуте — это уровень "пишет осмысленные предложения, держит тему 2-3 реплики". До GPT-3.5 далеко: нужно ~1B параметров и неделя H100. Что реально можно сделать сверх — увеличить до 100M, тренировать сутки. Это уже потребует терпения, но М4 потянет.
