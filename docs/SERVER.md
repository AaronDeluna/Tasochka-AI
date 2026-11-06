# Обучение на сервере с NVIDIA GPU

Код тот же, что на Mac. На сервере тренер сам выбирает PyTorch + CUDA.
Сначала пройди проверку на Mac ([HOWTO.md](HOWTO.md), шаги 2–3): на сервере
платишь за каждый час, и отлаживать там дорого.

## 0. Перенести проект

Запускается **на Mac**. `.venv`, данные и чекпойнты не копируем: данные
быстрее скачать прямо на сервере.

```bash
rsync -avz --exclude .venv --exclude datasets --exclude checkpoints --exclude .git ./ user@server-ip:~/tasochka/
```

## 1. Окружение (на сервере)

```bash
cd ~/tasochka && python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
```

Посмотри версию CUDA (`nvidia-smi`, строка `CUDA Version`) и поставь torch под
неё. Пример для CUDA 12.x:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install -r requirements/torch.txt
```

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Должно напечатать `True` и имя карты.

## 2. Данные (~13 GB)

```bash
python scripts/download_en.py --preset server
```

Скачивается 12 GB FineWeb-Edu и ~1.2 GB smol-smoltalk (весь датасет). Нужно
~30 GB свободного места: данные, кэш токенов и чекпойнты. Обрыв связи не страшен:
перезапусти ту же команду, и закачка продолжится. Закачку тоже лучше запускать
внутри `tmux` (см. ниже).

## 3. Обучение

Пресет `configs/server-125m.json`: ~125M параметров (класс GPT-2 small),
контекст 1024, 131k токенов на шаг, 20 000 шагов, ~2.6B токенов
(≈20 токенов на параметр по Chinchilla).

```bash
tmux new -s train
```

```bash
source .venv/bin/activate && python -m tasochka.train --config configs/server-125m.json 2>&1 | tee train.log
```

Отсоединиться от tmux: `Ctrl+B`, затем `D`. Вернуться: `tmux attach -t train`.

Первый запуск ~10–30 минут (зависит от числа ядер) учит токенизатор и
кодирует корпус. Потом идёт обучение. Как на Mac, прерванный прогон
продолжается той же командой.

**Память GPU.** Пресет рассчитан на 24 GB (`batch_size 16 × grad_accum 8`).
- Если упал по OOM: `--batch-size 8 --grad-accum 16` (эффективный батч тот же)
  или добавь `--grad-checkpoint`.
- На 40–80 GB можно `--batch-size 32 --grad-accum 4`: будет быстрее.
- Если `torch.compile` падает: поставь в пресете `"compile": false`. Без него
  обучение медленнее примерно в 1.3–1.5 раза, но работает.

**Какие цифры нормальные.** В первые сотни шагов loss падает с ~10.4 (ln 32000)
до ~6, к концу примерно до 3. Val loss идёт рядом с train. `gnorm` держится
в пределах ~0.2–1.5.

## Как чекпойнты защищены от потери

- **Каждые 1000 шагов** сохраняются веса, состояние оптимизатора и номер шага.
  Та же команда продолжает обучение ровно с этого места.
- **Запись атомарная.** Файл сначала пишется рядом, потом подменяет старый.
  Если процесс умрёт посреди записи, предыдущий чекпойнт останется целым.
- **Ctrl+C, `kill`, выключение сервера (SIGTERM) и закрытие SSH без tmux
  (SIGHUP)** перед выходом сохраняют текущий шаг. Ничего не теряется.
- **Каждые 5000 шагов** веса копируются в `snapshots/step_NNNNNN/`. Если
  обучение «взорвётся» (loss резко вырос и не вернулся), можно откатиться
  к снимку. Снимок — полноценный чекпойнт: его можно сразу открыть через
  `generate --checkpoint checkpoints/server-125m/snapshots/step_010000`.
- `weights_best.safetensors` хранит веса с лучшим val loss.

Всё это лежит на диске сервера. **Если удалить сервер, удалится и модель**,
поэтому время от времени забирай копию на Mac (шаг 4). Команда докачивает
только изменившееся.

## 4. Забрать модель на Mac

Запускается **на Mac**:

```bash
rsync -avz --exclude corpus_ids.bin --exclude optimizer.safetensors user@server-ip:~/tasochka/checkpoints/server-125m/ checkpoints/server-125m/
```

```bash
python -m tasochka.generate --checkpoint checkpoints/server-125m --chat
```

Веса в `safetensors` читаются обоими бэкендами (проверено: модель, обученная
на PyTorch, генерирует через MLX). На Mac модель запускается как обычно.

## 5. Не забыть

- Останови инстанс после обучения: деньги списываются, пока он включён.
- Перед запуском проверь: `nvidia-smi` видит карту, `df -h .` показывает ≥30 GB
  свободного места, `ls datasets/en/*/` показывает оба файла.
