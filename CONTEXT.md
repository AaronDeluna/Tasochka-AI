# Tasochka AI — context handoff

**Если ты Claude и читаешь это в новом чате — прочти ВНИМАТЕЛЬНО до конца, прежде чем отвечать пользователю.**

> **ОБНОВЛЕНИЕ 2026-07-02:** проект реорганизован и переработан. Код обучения
> с нуля переехал в `from-scratch/` (пакет `from-scratch/tasochka/`), дообучение —
> в `finetune/` (новый QLoRA-пайплайн). Актуальные команды — в `TRAINING_GUIDE.md`
> и `README.md`; структура ниже по тексту частично устарела. Репозиторий:
> github.com/AaronDeluna/Tasochka-AI (main + ветка training-improvements).

Это документ передачи контекста, который заменяет предыдущую переписку. Тут описано что сделано, что бежит, какие были решения и где мы остановились.

---

## Кто пользователь и что строим

- **Иван (zsh, Warp Terminal, Mac mini M4 16 GB)** — учится ML, делает свою языковую модель с нуля.
- **Цель**: маленький GPT-подобный ассистент на русском "Тасочка", обученный с нуля, работающий и на Mac (MLX), и на NVIDIA-сервере (PyTorch).
- **Уровень**: junior, нужно объяснять простыми словами и давать пошаговые команды копи-паст. Иногда косячит с активацией venv, путаницей путей, лишними кавычками. Будь терпеливым, давай команды одной строкой когда возможно.
- **Язык общения**: русский, неформальный.

---

## История проекта (краткий timeline)

1. **Стартовал** с Java + ND4J + CPU реализации char-level GPT. Был **сломанный backprop** в ND4J SameDiff (gradient не проходил через все блоки).
2. **Я починил Java версию**: переписал на SameDiff autograd, заработала, обучилась на ~30k шагов. Но Java/CPU = 400 мс/шаг, слишком медленно.
3. **Мигрировали на Python + MLX** (Apple GPU). Скорость в 30× быстрее. Char-level → BPE токенизатор (HF tokenizers, vocab 8000). Архитектура: RMSNorm + RoPE + SwiGLU (LLaMA-mini).
4. **Расширили датасет** через `download_datasets.py`: saiga_scored, grandmaster, gsm8k-ru, alpaca-cleaned-ru, saiga_preferences, Wikipedia RU 3GB, Veles-2.5, russian_dialogues.
5. **Persona seed**: создали `generate_seed.py` (имя "Тасочка", минимум 23 диалога × 3 повтора = 11 KB) и `generate_self_awareness.py` (437 диалогов о природе ИИ, 108 KB). Был период когда seed был раздут до 13MB с фактами/математикой — модель зубрила и проваливалась на CONTEXT-тестах. **Сейчас seed минимальный**, только имя + identity.
6. **Несколько прогонов train**:
   - Mac char-level 49M, 30k шагов → gibberish с осмысленными моментами
   - Mac BPE 49M, 100k шагов → val 1.37, базовая речь работает, факты выдумывает
   - Mac BPE 200M, 150k шагов (с раздутым seed) → CONTEXT-failure (модель игнорирует промпт, отвечает заученным)
   - Mac BPE 200M, ~30k шагов (с минимальным seed) → не закончился, прервали в пользу серверного запуска
7. **Портировали на PyTorch** для запуска на NVIDIA сервере. Backend switch: `--backend mlx|torch` или auto-detect. Чекпойнты в safetensors **совместимы между MLX и PyTorch**.

---

## ЧТО БЕЖИТ ПРЯМО СЕЙЧАС

**На NVIDIA-сервере A5000 24GB (через Selectel/Timeweb, IP `194.67.119.224`):**

```bash
tmux session 'train'
python -m tasochka.train --backend torch \
  --max-steps 200000 \
  --batch-size 12 \
  --context-length 1024 \
  --max-dataset-chars 5000000000 \
  --lr 2e-4 \
  --warmup-steps 3000 \
  --min-lr-ratio 0.05 \
  --bf16 \
  --compile
```

**Модель:**
- 207M параметров
- 20 слоёв × dim=896 × heads=14 × ff=2400
- ctx=1024 BPE токенов
- RMSNorm + RoPE + SwiGLU

**По датасету:**
- ~6.4 GB чистого русского текста (Wikipedia 5GB + chat 1.4GB)
- ~1.4B BPE токенов после кодирования
- 200k × 12 × 1024 = 2.46B token-positions → **1.75 эпохи**, 62% от Chinchilla

**Тайминги:**
- Encoding: ~45 минут (одноразово, кэш `corpus_ids.bin`)
- Train 200k шагов: **~25-28 часов**
- Стоимость: ~57₽/час × 28ч = **~1600₽**

**Окружение сервера:**
- Ubuntu 26.04 LTS, kernel 7.0
- Python 3.14 системный, **в venv через uv поставлен Python 3.12.13** (PyTorch wheels не было под 3.14)
- PyTorch 2.6.0+cu124
- NVIDIA Driver 595.71.05, CUDA 13.2 (backward compatible с cu124 wheels)
- venv: `/root/tasochka/.venv`
- uv: `/root/.local/bin/uv` (добавлен в PATH через `~/.bashrc`)

---

## Структура файлов

```
/Users/ivanmilovanov/Desktop/Tasochka AI/   (Mac)
└── /root/tasochka/                          (сервер — копия через rsync)

├── tasochka/
│   ├── _backend.py        # dispatcher: парсит --backend, auto-detect
│   ├── train.py           # тонкий dispatcher для train (mlx или torch)
│   ├── generate.py        # тонкий dispatcher для generate
│   │
│   ├── model_mlx.py       # MLX MiniGPT (для Mac)
│   ├── train_mlx.py       # MLX training loop
│   ├── generate_mlx.py    # MLX generation
│   │
│   ├── model_torch.py     # PyTorch MiniGPT (для сервера)
│   ├── train_torch.py     # PyTorch training loop (с --bf16 --compile)
│   ├── generate_torch.py  # PyTorch generation
│   │
│   ├── data.py            # JSONL/TXT стрим, BPE encoding, memmap токенов
│   ├── tokenizer.py       # HF BPE tokenizer wrapper
│   ├── eval.py            # 43 промпта вне seed для честной оценки
│   └── server.py          # FastAPI с Ollama-совместимым /api/chat
│
├── russian-train-dataset/         # датасет, на сервере и Mac
│   ├── wikipedia_ru.txt           # 5 GB
│   ├── saiga_chat.txt             # 167 MB
│   ├── grandmaster_chat.txt       # 315 MB
│   ├── saiga_prefs_chat.txt       # 169 MB
│   ├── alpaca_ru_chat.txt         # 73 MB
│   ├── ru_turbo_alpaca_chat.txt   # 35 MB (только Mac)
│   ├── ru_instruct_chat.txt       # 1.6 GB (только Mac!)
│   ├── gsm8k_chat.txt             # 6.7 MB
│   ├── veles_chat.txt             # 130 MB
│   ├── russian_dialogues_chat.txt # 545 MB
│   ├── persona_seed.txt           # 11 KB
│   └── self_awareness.txt         # 108 KB
│
├── checkpoints/tasochka/
│   ├── tokenizer.json             # BPE
│   ├── config.json                # архитектура
│   ├── weights.safetensors        # совместимо MLX↔PyTorch
│   ├── corpus_ids.bin             # ~5 GB memmap токенов (можно не копировать)
│   └── corpus_ids.meta
│
├── download_datasets.py           # качает датасеты с HuggingFace
├── generate_seed.py               # генерит persona_seed.txt
├── generate_self_awareness.py     # генерит self_awareness.txt
├── requirements.txt               # MLX базовый
├── requirements-torch.txt         # PyTorch для сервера
├── requirements-server.txt        # FastAPI для веб-чата
├── setup.sh                       # один раз для venv
├── index.html                     # веб-чат, говорит по Ollama API
├── RUN_MAC.md                     # инструкции для Mac
├── RUN_SERVER.md                  # инструкции для сервера
├── README.md
└── CONTEXT.md                     # ЭТОТ ФАЙЛ
```

---

## Ключевые архитектурные решения (зачем именно так)

### Backend switch (MLX + PyTorch одновременно)
- Mac: MLX 2× быстрее PyTorch+MPS, оставляем для Mac.
- Сервер: PyTorch обязателен (NVIDIA = CUDA).
- Дispatcher `_backend.py` парсит `--backend mlx|torch`, иначе auto-detect.
- safetensors веса работают на обоих → можно перенести checkpoint между.

### Архитектура модели (LLaMA-style mini)
- **RoPE** (Rotary Position Embedding) вместо learned positional. Лучше extrapolation.
- **RMSNorm** вместо LayerNorm. Стабильнее в FP16/BF16.
- **SwiGLU** в FF вместо GELU. Лучше учится при том же бюджете.
- **Pre-norm** residual connections.
- **Causal attention** через `scaled_dot_product_attention` (Flash Attention автоматом на CUDA, MLX `mx.fast.scaled_dot_product_attention` на Mac).
- Vocabulary 8000 BPE токенов через HF tokenizers (ByteLevel).

### Data loading
- `iter_cleaned_chunks` — стрим из .txt файлов в директории, round-robin интерливит.
- Маленькие файлы (<100 MB) **циклятся** при EOF чтобы persona_seed появлялся по всему training stream.
- Остановка когда все большие файлы (>100 MB) выработаны.
- Cleaning: разрешает кириллицу, латиницу, цифры, base punctuation. Жёсткие control codes выбрасывает.
- Encoding пишет в `corpus_ids.bin` (int32 memmap, ~5 GB), потом читается без загрузки в RAM.

### Persona seed философия
- **Минимальный seed**: только то что модель *не может узнать из общего корпуса*.
- Имя "Тасочка" — выдуманное слово, нет нигде в интернете. **Только seed**.
- "Я ИИ-модель" — есть в saiga, но без конкретной идентификации.
- Refusals на выдуманные имена (Балалайкин, Зюзин-Хрюндер) — учит **паттерну** отказа.
- **НЕТ** в seed: столицы, авторы, таблица умножения. Это читы.

### Sampling (generate)
- Temperature (default 0.8)
- Top-k (50) + top-p / nucleus (0.92)
- **Repetition penalty** (1.18) на последние 128 токенов
- **No-repeat n-gram** (3) — блокирует дословные повторы 3-грамм
- **UTF-8-safe streaming**: если декод даёт U+FFFD (replacement char) в конце — придерживает кусок до следующего токена.
- Stop strings: `</assistant>`, `<user>`, `<|endoftext|>`, `</s>`.

---

## Что узнали по ходу (ошибки которых избегать)

1. **gather backward в nd4j M2.1 сломан** — пришлось делать через one-hot @ matmul. (Уже не актуально, но для понимания почему MLX выбран.)
2. **Apple обновил Command Line Tools → venv с системным python3 ломается.** Симптом: `.venv/bin/python` → `python3` → не существует. Лечение: `rm -rf .venv && /usr/bin/python3 -m venv .venv && pip install ...`. Или ставить через `uv python install 3.12 && uv venv --python 3.12`.
3. **Ubuntu 26.04 + Python 3.14 — у PyTorch нет wheels.** На сервере нужно ставить Python 3.12 через `uv python install 3.12`, иначе `pip install torch` не находит совместимую версию.
4. **PEP 668 externally-managed-environment**: на новой Ubuntu системный pip заблокирован. Решение: либо venv с pip (но `uv venv` создаёт venv БЕЗ pip), либо использовать `uv pip install` (но uv нужен в PATH).
5. **Warp Terminal — каждая команда в новой подсессии.** `source .venv/bin/activate` не сохраняется между блоками. Решение: использовать абсолютный путь `./.venv/bin/python ...`.
6. **Persona seed × 100 повторов → CONTEXT failure.** Модель зазубрила и игнорировала промпт. Урок: seed должен быть микроскопическим (минимум 3 повтора, ~11 KB файл).
7. **`set -g mouse on` в tmux** — иначе колесо мыши шлёт arrow-key escape sequences.

---

## Текущие выполненные задачи (40 штук)

Все архитектурные, кодовые и инфраструктурные задачи выполнены. Полный список с # 1 до # 40 был в task tracker предыдущего чата:
1-12: Java→Python+MLX миграция, FastAPI, начальная обвязка
13-19: BPE, LLaMA-mini архитектура, smoke-тесты
20-25: Interleave, cosine LR, streaming memmap, repetition penalty, UTF-8 buffer
26-31: Расширение датасетов, persona seed (раздут потом урезан), self-awareness, eval вне seed
32: Wikipedia RU + Veles + Russian Dialogues
33-37: MLX→PyTorch портирование, backend dispatcher
38-39: requirements-torch, smoke
40: Финальный план A5000 200k шагов ctx=1024

**Никаких pending tasks** — train идёт на сервере.

---

## Что делать дальше (как пользователю продолжать)

### Если train ещё идёт на сервере
Ждать. Подключаться по SSH к `root@194.67.119.224`, `tmux attach -t train`. Смотреть прогресс. Не закрывать tmux окно.

### После завершения train (через ~28 часов от старта 25 мая 21:00)
1. Скопировать чекпойнт с сервера на Mac (**только weights.safetensors, config.json, tokenizer.json**, без corpus_ids.bin):
   ```bash
   # На Mac (НЕ на сервере):
   rsync -avz --progress \
     --exclude='corpus_ids.bin' \
     --exclude='corpus_ids.meta' \
     root@194.67.119.224:/root/tasochka/checkpoints/tasochka/ \
     "/Users/ivanmilovanov/Desktop/Tasochka AI/checkpoints/tasochka_server/"
   ```
2. **ВЫКЛЮЧИТЬ СЕРВЕР** в личном кабинете провайдера (иначе деньги списываются).
3. Запустить eval:
   ```bash
   cd "/Users/ivanmilovanov/Desktop/Tasochka AI"
   # Поменять checkpoints/tasochka на checkpoints/tasochka_server в командах, или временно переименовать
   ./.venv/bin/python -m tasochka.eval > eval_output_server.txt
   ```
4. Запустить чат:
   ```bash
   ./.venv/bin/python -m tasochka.generate --chat
   ```

### Если хочется продолжить тренировку (continue training)
Запустить ту же команду что и был train — auto-resume из чекпойнта. Учти что cosine LR в финале опустился до 1e-5; для дальнейшего обучения используй `--lr 5e-5 --warmup-steps 100`.

---

## Команды-цитата для быстрого доступа

### Train на Mac (MLX)
```bash
cd "/Users/ivanmilovanov/Desktop/Tasochka AI"
./.venv/bin/python -m tasochka.train --max-steps 150000 --batch-size 5 --context-length 512 --max-dataset-chars 5000000000 --lr 2.5e-4 --warmup-steps 2000 --min-lr-ratio 0.1
```

### Train на сервере (PyTorch)
```bash
cd /root/tasochka
source .venv/bin/activate
python -m tasochka.train --backend torch --max-steps 200000 --batch-size 12 --context-length 1024 --max-dataset-chars 5000000000 --lr 2e-4 --warmup-steps 3000 --min-lr-ratio 0.05 --bf16 --compile
```

### Чат
```bash
./.venv/bin/python -m tasochka.generate --chat
```

### Eval
```bash
./.venv/bin/python -m tasochka.eval > eval_output.txt
```

---

## Если возникает новый вопрос

- Если просит запустить что-то локально — **давай команду одной строкой**, без \ переносов (Warp может ломать).
- Если "command not found: python" — используй `./.venv/bin/python ...` напрямую.
- Если "ModuleNotFoundError" — скорее всего venv сломан, пересоздать.
- Если запутался в Warp терминале — учти что каждый блок в отдельной подсессии, source не сохраняется.

---

## Финальная заметка

Это **учебный проект**. Цель не "сделать ChatGPT", а понять как языковые модели устроены изнутри. 200M на 1.75 эпохах ≈ GPT-2 medium по качеству. Реалистичный потолок для домашнего обучения с нуля. Дальше нужно либо железо за миллионы $, либо fine-tune готовой Llama/Saiga (другая задача).

Будь честным про ограничения. Не обещай ChatGPT-уровень. Хвали конкретные успехи (persona, refusals, базовые факты после Wikipedia).
