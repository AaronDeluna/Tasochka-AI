# Гайд по обучению на Mac mini M4 16GB

Все команды — одной строкой, копируй и вставляй в Warp целиком.
**Запускать из корня проекта** (`./.venv/bin/python ...` — venv лежит в корне).

> 📘 Подробный разбор всех параметров простыми словами, данные под 1B и как
> общаться с моделью — в [HOWTO.md](HOWTO.md). Здесь — краткая шпаргалка команд.

Код — пакет `tasochka/`, данные — `datasets/` (по подпапке на источник, читаются
рекурсивно), чекпойнты — `checkpoints/`.

---

## Своя модель с нуля

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
чекпоинт: `rm -rf checkpoints/tasochka` (токенизатор и кэш токенов
можно оставить, они пересоздадутся только при смене датасета).

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
cd "/Users/ivanmilovanov/Desktop/Tasochka AI" && ./.venv/bin/python -m tasochka.generate --chat
cd "/Users/ivanmilovanov/Desktop/Tasochka AI" && ./.venv/bin/python -m tasochka.eval > eval_output.txt
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
