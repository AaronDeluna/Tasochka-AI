# Запуск на Mac (Apple Silicon — M1/M2/M3/M4)

Tasochka AI бежит на встроенной GPU Mac через MLX. Быстро, нативно, ничего сверх macOS не нужно.

---

## 0. Один раз — установка

Открой Terminal (или Warp/iTerm). Выполни **каждую команду по очереди**:

```bash
cd "/Users/ivanmilovanov/Desktop/Tasochka AI"
```

```bash
/usr/bin/python3 -m venv .venv
```

```bash
./.venv/bin/pip install --upgrade pip
```

```bash
./.venv/bin/pip install mlx numpy tokenizers datasets
```

Установка идёт ~3-5 минут. После этого .venv готов.

---

## 1. Подготовка к долгому train

Перед каждым запуском train на сутки+:

**Запрети маку засыпать** (запусти в отдельном окне терминала, оставь работать):
```bash
caffeinate -dimsu
```

Это окно теперь "висит" — ничего не делай в нём пока train идёт. После train — `Ctrl+C` чтобы выключить caffeinate.

**Закрой жрущие приложения**: Chrome, Slack, IDE. Освободи RAM.

**В System Settings:**
- Battery / Energy → "Prevent automatic sleeping" → ON
- General → Software Update → отключи Auto-Update на время train

---

## 2. Запуск тренировки

```bash
cd "/Users/ivanmilovanov/Desktop/Tasochka AI"
```

```bash
./.venv/bin/python -m tasochka.train --max-steps 150000 --batch-size 5 --context-length 512 --max-dataset-chars 5000000000 --lr 2.5e-4 --warmup-steps 2000 --min-lr-ratio 0.1
```

**Что это запустит:**
- 200M параметров (20 слоёв)
- Контекст 512 BPE токенов
- 150 000 шагов обучения
- Использует **MLX → Apple GPU** автоматически

**Сколько по времени:** ~3.5 суток на M4 16 GB.

**Что увидишь:**
```
encoded in 40m: 5G chars → 1.5B tokens
init forward loss=9.12 (random baseline ~8.99)
step 1/150000 (0.0%) loss 9.11 ... 2000 ms/step ETA 83h
step 20/150000 ...
...
```

Если хочешь чтобы не зависело от окна терминала — добавь в начале `nohup` и в конце `> train.log 2>&1 &`:

```bash
nohup ./.venv/bin/python -m tasochka.train --max-steps 150000 --batch-size 5 --context-length 512 --max-dataset-chars 5000000000 --lr 2.5e-4 --warmup-steps 2000 --min-lr-ratio 0.1 > train.log 2>&1 &
```

Прогресс смотреть:
```bash
tail -f train.log
```

Остановить:
```bash
ps aux | grep tasochka.train
kill <pid из вывода выше>
```

---

## 3. Общение с моделью

После train закончится (или прервал Ctrl+C — чекпойнт сохранён):

```bash
cd "/Users/ivanmilovanov/Desktop/Tasochka AI"
./.venv/bin/python -m tasochka.generate --chat
```

Внутри:
```
> Привет
[ответ модели]
> Что такое Москва?
[ответ]
> exit
```

---

## 4. Если что-то не так

**"command not found: python"**
→ Используй `./.venv/bin/python` (точно как в командах выше)

**"ModuleNotFoundError: No module named 'mlx'"**
→ venv сломан (Apple обновил CommandLineTools). Пересоздай:
```bash
cd "/Users/ivanmilovanov/Desktop/Tasochka AI"
rm -rf .venv
/usr/bin/python3 -m venv .venv
./.venv/bin/pip install mlx numpy tokenizers datasets
```

**Train упал на середине**
→ Чекпойнты каждые 500 шагов. Просто запусти ту же команду — продолжит автоматически.

**Хочу начать с нуля**
```bash
rm -rf checkpoints/tasochka
```
Потом запусти train заново.

**Хочу пообщаться пока train ещё идёт**
→ Не рекомендую — оба процесса лезут в GPU, замедлит train в 1.5-2 раза. Если очень надо — открой второй терминал и запусти `--chat`. Чекпойнт автоматически берётся последний сохранённый.

---

## 5. Команда быстрого старта (одной строкой)

Скопируй и вставь — всё что нужно в одну операцию:

```bash
cd "/Users/ivanmilovanov/Desktop/Tasochka AI" && ./.venv/bin/python -m tasochka.train --max-steps 150000 --batch-size 5 --context-length 512 --max-dataset-chars 5000000000 --lr 2.5e-4 --warmup-steps 2000 --min-lr-ratio 0.1
```
