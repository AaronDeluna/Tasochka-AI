# Гайд по обучению на Mac mini M4 16GB

Все команды — одной строкой, копируй и вставляй в Warp целиком.

Есть два пути. **Если цель — «поговорить с моделью по-русски на нормальные темы», иди по Пути 1.**

| | Путь 1: QLoRA (рекомендую) | Путь 2: с нуля |
|---|---|---|
| Что это | Дообучаем готовую Gemma 3 4B под персону Тасочки | Тренируем свой мини-GPT со случайных весов |
| Качество речи | Уровень нормального ассистента | «Пишет связные предложения», не более |
| Время | ~2–4 часа | от ночи до суток |
| Зачем | Реально пользоваться | Учиться, понимать LLM изнутри |

---

## Путь 1: QLoRA — русскоязычный ассистент за вечер

Это тот же подход, которым китайские лаборатории (DeepSeek, Qwen) экономят ресурсы:
базовая модель в **4-bit квантовании** (~2.7 GB вместо 8 GB), поверх неё учится
маленький **LoRA-адаптер**. Память свободна → можно длинный контекст и все слои.

### Шаг 1. Собрать датасет (один раз, ~5 минут)

```bash
cd "/Users/ivanmilovanov/Desktop/Tasochka AI" && ./.venv/bin/python finetune/build_chat_dataset.py
```

Что внутри: ~5500 реальных русских диалогов (saiga GPT-4 диалоги с оценкой ≥7,
grandmaster, живая разговорная речь) + persona Тасочки. Всё отфильтровано:
только полные диалоги, кириллица, без «я Сайга/GPT/YandexGPT» (конфликты с персоной),
дедупликация. Принцип Qwen/DeepSeek: **маленький чистый SFT-датасет бьёт большой грязный**.

### Шаг 2. Обучить (~2–4 часа, смотри ETA в логе)

```bash
cd "/Users/ivanmilovanov/Desktop/Tasochka AI" && bash finetune/train_qlora.sh
```

При первом запуске сама скачается `mlx-community/gemma-3-4b-it-qat-4bit` (~2.5 GB).
QAT-версия — Google квантовал модель с дообучением, качество почти как у полной.

Все настройки в [finetune/qlora_config.yaml](finetune/qlora_config.yaml) — файл с комментариями, что каждая строка делает.

### Шаг 3. Проверить

```bash
bash finetune/chat.sh "Привет! Как тебя зовут?"
bash finetune/chat.sh "Расскажи, как прошёл бы твой идеальный день"
bash finetune/chat.sh "Ты Gemma?"
```

### Шаг 4. Слить в одну модель (опционально)

```bash
bash finetune/fuse.sh
```

### Что крутить, если что-то не так

| Симптом | Что менять (в qlora_config.yaml) |
|---|---|
| Всё ещё говорит «я Gemma» | `iters: 3000` или пересобери датасет с `--persona-mult 3` |
| Отвечает persona-фразами на любой вопрос (переучилась) | `iters: 1000-1500`, либо датасет с `--persona-mult 1` |
| Не хватает памяти (swap, всё тормозит) | `batch_size: 1` + `grad_accumulation_steps: 8`; закрой браузер |
| Хочется быстрее (жертвуя памятью) | `grad_checkpoint: false` (если памяти хватает) |
| Ответы слишком «сухие»/шаблонные | `lora_parameters.rank: 32` + датасет побольше: в build_chat_dataset.py увеличь лимиты в SOURCES |
| val loss растёт с какого-то шага | Это переобучение — возьми адаптер из `finetune/adapters_qlora/` с шага до роста (они сохраняются каждые 250) |

Продолжить обучение с места остановки:

```bash
bash finetune/train_qlora.sh --resume-adapter-file finetune/adapters_qlora/adapters.safetensors
```

Хочешь попробовать другую базу («китаянку»): в `qlora_config.yaml` поменяй
`model:` на `mlx-community/Qwen3-4B-Instruct-2507-4bit` — Qwen отлично говорит
по-русски. Датасет и остальное не трогай.

---

## Путь 2: своя модель с нуля (учебный)

Тренер переписан по лучшим практикам LLaMA/DeepSeek/Qwen. Что нового:

| Улучшение | Что даёт |
|---|---|
| `--bf16` | **~2x скорость, 2x меньше памяти** (bfloat16 на Apple GPU) |
| `mx.compile` | Train step компилируется в один Metal-граф (автоматически, ещё ~15-30%) |
| `--num-kv-heads N` | GQA как в Qwen/LLaMA-3: меньше KV-голов → меньше параметров и памяти |
| `--tie-embeddings` | LM-голова = матрица эмбеддингов, экономит ~7M параметров |
| `--grad-accum N` | Эффективный батч больше без роста памяти |
| Точный resume | Состояние оптимизатора и номер шага сохраняются — прервал и продолжил без потерь |
| `weights_best.safetensors` | Автоматически хранится лучший по val loss чекпоинт |
| AdamW betas (0.9, 0.95), wd 0.1, init 0.02/√(2L) | Стандарт GPT-3/LLaMA — стабильнее и чуть лучше сходится |

**ВАЖНО:** флаги `--tie-embeddings` и `--num-kv-heads` меняют архитектуру,
поэтому действуют только на НОВУЮ модель. Для нового прогона удали старый
чекпоинт: `rm -rf checkpoints/tasochka` (токенизатор и кэш токенов можно
оставить, они пересоздадутся только при смене датасета).

### Рекомендуемый конфиг: ~80M, вечер обучения

```bash
cd "/Users/ivanmilovanov/Desktop/Tasochka AI" && ./.venv/bin/python -m tasochka.train --bf16 --tie-embeddings --embedding-dim 768 --num-layers 12 --num-heads 12 --num-kv-heads 4 --feed-forward-dim 2048 --context-length 512 --batch-size 8 --grad-accum 2 --max-steps 30000 --lr 3e-4 --warmup-steps 1000 --min-lr-ratio 0.1 --max-dataset-chars 2000000000 --val-every 500 --checkpoint-every 1000
```

### Конфиг побольше: ~190M, ночь-сутки

```bash
cd "/Users/ivanmilovanov/Desktop/Tasochka AI" && ./.venv/bin/python -m tasochka.train --bf16 --tie-embeddings --embedding-dim 1024 --num-layers 16 --num-heads 16 --num-kv-heads 4 --feed-forward-dim 2816 --context-length 512 --batch-size 4 --grad-accum 4 --max-steps 60000 --lr 2.5e-4 --warmup-steps 2000 --min-lr-ratio 0.1 --max-dataset-chars 5000000000 --val-every 500 --checkpoint-every 1000
```

Прервать можно в любой момент (Ctrl+C) — повторный запуск той же команды
продолжит ровно с последнего чекпоинта (шаг, оптимизатор, LR — всё восстановится).

### Пообщаться / оценить

```bash
./.venv/bin/python -m tasochka.generate --chat
./.venv/bin/python -m tasochka.eval > eval_output.txt
```

Для чата с лучшим (а не последним) чекпоинтом: скопируй
`weights_best.safetensors` поверх `weights.safetensors` в копии папки чекпоинта.

### Что крутить

| Симптом | Что менять |
|---|---|
| Не хватает памяти | `--batch-size` меньше + `--grad-accum` больше (эффективный батч тот же) |
| loss скачет / gnorm большой | `--lr` в 1.5-2 раза ниже, `--warmup-steps` больше |
| Медленно (смотри tok/s в логе) | Проверь что есть `--bf16`; уменьши `--context-length` |
| Модель тупая после обучения | Больше шагов и данных: правило Chinchilla ~20 токенов на параметр |

---

## Память: как не упереться в 16 GB

- Закрой Chrome/Safari перед долгим обучением — они легко съедают 4-6 GB.
- Следи за памятью: `sudo memory_pressure -Q` или просто Activity Monitor.
- Если система ушла в swap (диск шуршит, всё тормозит) — обучение станет в
  10 раз медленнее. Лучше меньший батч, чем swap.
- QLoRA-путь при дефолтном конфиге ест ~6-8 GB — безопасно.
