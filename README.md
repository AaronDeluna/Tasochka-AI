# Tasochka AI

Свой ИИ-ассистент общего назначения, обученный **с нуля**: LLaMA-подобный
трансформер (BPE · RMSNorm · RoPE · SwiGLU · GQA · tied embeddings · bf16).
Сейчас целимся в **английский** язык. Порядок работы: короткая проверка на Mac
(MLX), затем полноценный прогон на сервере с NVIDIA GPU (PyTorch). Код и флаги
у обоих бэкендов общие.

**Пошаговая инструкция: [docs/HOWTO.md](docs/HOWTO.md)** · сервер: [docs/SERVER.md](docs/SERVER.md)

## Быстрый старт (Mac)

Все команды запускаются из корня проекта.

```bash
./setup.sh && source .venv/bin/activate
```

```bash
python scripts/download_en.py --preset mac
```

```bash
python -m tasochka.train --config configs/mac-smoke.json
```

```bash
python -m tasochka.generate --checkpoint checkpoints/mac-smoke --chat
```

Smoke-тест идёт ~1 минуту и проверяет конвейер целиком. Часовая проверка запускается
пресетом `configs/mac-1h.json`.

## Структура

```
tasochka/            пакет: модель, обучение, генерация, сервер (MLX + PyTorch)
  train.py             точка входа обучения → train_mlx.py / train_torch.py
  generate.py          генерация и чат → generate_mlx.py / generate_torch.py
  model_*.py           архитектура (одинаковая на обоих бэкендах)
  data.py              потоковое чтение, очистка и смешивание корпуса
  tokenizer.py         BPE (HuggingFace tokenizers)
  config.py            загрузка JSON-пресетов (--config)
  eval.py              фиксированный набор промптов (--lang en|ru)
  server.py            HTTP API по протоколу Ollama для web/index.html
configs/             пресеты обучения: mac-smoke · mac-1h · server-125m
scripts/
  download_en.py       английский датасет: FineWeb-Edu + smol-smoltalk
  download_ru.py       русские датасеты (прошлая версия проекта)
  ru_*.py              русская персона «Тасочка» (seed / self-awareness)
web/index.html       веб-чат
requirements/        base.txt (Mac/MLX) · torch.txt (сервер) · server.txt (веб-API)
docs/                HOWTO.md · SERVER.md · archive/ (документация русской версии)
datasets/            данные, не в git: en/<источник>/*.txt, ru/<источник>/*.txt
checkpoints/         чекпойнты, не в git: checkpoints/<имя-прогона>/
```

## Ожидания по качеству

Модель на 16M параметров после часа на MacBook пишет грамматичный, но
бессвязный английский. Её задача — подтвердить, что обучение идёт правильно.
Модель на ~125M параметров после сервера держит тему и отвечает в формате
ассистента, но фактов почти не знает: это уровень GPT-2 small. Чтобы
приблизиться к ChatGPT, понадобятся модели на 1B+ параметров, сотни миллиардов
токенов и отдельные стадии SFT/RLHF. Текущий конвейер — фундамент для этого.
