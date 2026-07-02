# Запуск на сервере с NVIDIA GPU

Tasochka AI бежит на NVIDIA GPU через PyTorch + CUDA. **В 5-30 раз быстрее Mac** в зависимости от карты.

Подходит для серверов с A4000, A5000, A100, H100 и др.

---

## Что тебе нужно

1. **Сервер с NVIDIA GPU и CUDA** (от любого облачного провайдера: timeweb, selectel, hetzner, vast.ai и т.д.)
2. **SSH доступ** к серверу
3. **Python 3.9+** (обычно уже стоит на Ubuntu)
4. **NVIDIA драйверы и CUDA toolkit** (обычно уже есть в готовых GPU-образах)

---

## 0. Один раз — установка на сервере

После SSH в сервер:

```bash
# Создаём папку проекта
mkdir -p ~/tasochka && cd ~/tasochka
```

Скопируй сюда **всю папку Tasochka AI** с Mac (через `scp` или `rsync`):

```bash
# Это запускается НА MAC, не на сервере:
scp -r "/Users/ivanmilovanov/Desktop/Tasochka AI/" user@server-ip:/home/user/tasochka/
```

**Не копируй `.venv`** — он Mac-only. Лучше предварительно удали:
```bash
# На Mac перед копированием:
rm -rf "/Users/ivanmilovanov/Desktop/Tasochka AI/.venv"
```

На сервере создай свежий venv и поставь PyTorch:

```bash
cd ~/tasochka
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

**Выбери правильную версию torch под свой CUDA:**

```bash
# Проверь версию CUDA на сервере:
nvidia-smi
```

Найди строку `CUDA Version: XX.X`. Дальше:

**Для CUDA 12.x:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Для CUDA 11.8:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Затем остальные зависимости:
```bash
pip install numpy tokenizers datasets safetensors
```

Проверка:
```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0))"
```

Должно вывести что-то типа:
```
cuda: True device: NVIDIA RTX A4000
```

---

## 1. Подготовка датасета

Если копировал всю папку — `russian-train-dataset/` уже на сервере (~8 GB).

Если нет, скачай отдельно:
```bash
cd ~/tasochka
python download_datasets.py --wiki-max-chars 3000000000
```

(Займёт 1-3 часа в зависимости от сети сервера.)

---

## 2. Запуск тренировки

```bash
cd ~/tasochka
source .venv/bin/activate
```

```bash
python -m tasochka.train --backend torch --max-steps 150000 --batch-size 16 --context-length 512 --max-dataset-chars 5000000000 --lr 2.5e-4 --warmup-steps 2000 --min-lr-ratio 0.1 --bf16 --compile
```

**Что отличается от Mac команды:**
- `--backend torch` — явно PyTorch (на самом деле auto-detect его и сам выберет)
- `--batch-size 16` — больше памяти GPU, можно больше батч
- `--bf16` — mixed precision на CUDA даёт ~2× ускорение
- `--compile` — torch.compile (+1.5×, первый шаг будет медленный)

**Сколько по времени:**
- A4000 16GB: ~6-8 часов
- A5000 24GB: ~4-6 часов
- A100 80GB: ~2-3 часа

(на тех же 150k шагах что Mac делает за 3.5 суток)

---

## 3. Запуск в фоне (важно для аренды)

Сервер арендован — деньги списываются пока он работает. Чтобы не зависеть от SSH сессии:

**Вариант с tmux** (рекомендую):
```bash
tmux new -s train
# внутри tmux — запусти команду train
# отсоединиться: Ctrl+B, потом D
# вернуться позже: tmux attach -t train
```

**Вариант с nohup:**
```bash
nohup python -m tasochka.train --backend torch --max-steps 150000 --batch-size 16 --context-length 512 --max-dataset-chars 5000000000 --lr 2.5e-4 --warmup-steps 2000 --min-lr-ratio 0.1 --bf16 --compile > train.log 2>&1 &
```

Смотреть прогресс:
```bash
tail -f train.log
```

---

## 4. Перенос чекпойнта обратно на Mac

После train на сервере скачай результат на Mac:

```bash
# Запускается НА MAC:
rsync -avz user@server-ip:/home/user/tasochka/checkpoints/tasochka/ "/Users/ivanmilovanov/Desktop/Tasochka AI/checkpoints/tasochka/"
```

Веса в формате `safetensors` — **работают и в MLX и в PyTorch**. На Mac запустишь генерацию через MLX как обычно.

---

## 5. Общение с моделью на сервере

```bash
cd ~/tasochka
source .venv/bin/activate
python -m tasochka.generate --chat
```

---

## 6. Если что-то не так

**"torch.cuda.is_available() → False"**
→ CUDA не настроена или драйверы не подходят. Проверь `nvidia-smi`. Возможно нужен правильный CUDA toolkit под версию драйвера.

**OOM (Out of memory)**
→ Уменьши `--batch-size`. На 16GB карте: попробуй 12, 8, 4.

**"ModuleNotFoundError: No module named 'torch'"**
→ Не активирован venv. Сделай `source .venv/bin/activate`.

**Не хочет компилироваться**
→ Убери `--compile`. Это опционально, без него тоже быстро.

---

## 7. Один раз перед запуском — финальный чек

```bash
# Должны быть все ОК:
ls russian-train-dataset/*.txt | wc -l    # >=10 файлов
nvidia-smi                                  # видит GPU
source .venv/bin/activate && python -c "import torch; print(torch.cuda.is_available())"  # True
df -h .                                     # минимум 30 GB свободно
```

Если все ОК — запускай train.

---

## 8. Сколько денег уйдёт

Примерные цены (Selectel/timeweb весна 2026):

| GPU | ₽/час | Train 150k шагов | Итого |
|---|---|---|---|
| A4000 16GB | 37 | ~6-8 ч | **~250-300₽** |
| A5000 24GB | 80 | ~4-6 ч | **~350-500₽** |
| A100 80GB | 200 | ~2-3 ч | **~500-700₽** |

**Дешевле A4000 чем электричество M4 за 3.5 суток.** И быстрее в десятки раз.

**Не забудь остановить инстанс после train.** Списание идёт пока сервер живой, даже если ничего не считает.
