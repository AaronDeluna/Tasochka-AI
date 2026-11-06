#!/usr/bin/env bash
# Забирает чекпойнт с сервера на Mac по кругу, пока идёт обучение.
#
#   scripts/sync_checkpoint.sh root@89.104.69.161 server-125m          # каждые 10 мин
#   scripts/sync_checkpoint.sh root@89.104.69.161 server-125m 300      # каждые 5 мин
#   scripts/sync_checkpoint.sh root@89.104.69.161 server-125m --once   # один раз
#
# Что делает каждый проход:
#   * rsync checkpoints/<run>/ с сервера: веса, оптимизатор, снимки — всё, что
#     нужно, чтобы продолжить обучение, если сервер умрёт (без кэша токенов,
#     его можно пересобрать). Копируются только изменившиеся файлы.
#   * для каждого нового шага кладёт копию весов в checkpoints/<run>/history/step_NNNNNN/
#     (готовый чекпойнт для generate/eval — сравнивать, как модель умнеет).
#     Копия через APFS-клон (cp -c): мгновенно и без лишнего места, пока файл не меняется.
#
# Mac не уснёт, пока скрипт работает (caffeinate), но крышку не закрывай —
# MacBook с закрытой крышкой засыпает всё равно. Остановить: Ctrl+C.
set -euo pipefail

HOST="${1:?usage: sync_checkpoint.sh user@host run_name [interval_sec|--once]}"
RUN="${2:?usage: sync_checkpoint.sh user@host run_name [interval_sec|--once]}"
MODE="${3:-600}"

cd "$(dirname "$0")/.."
LOCAL="checkpoints/$RUN"
mkdir -p "$LOCAL/history"

sync_once() {
  # -W (whole file) and no -z: weights change completely every checkpoint and
  # random floats don't compress, so delta search and compression only burn CPU.
  # weights_best.safetensors is skipped: while val keeps improving it is a copy
  # of the latest weights (history/ keeps every step anyway).
  if ! rsync -a -W --partial --exclude corpus_ids.bin --exclude corpus_ids.meta \
       --exclude weights_best.safetensors --exclude 'history/' --exclude '.tmp-*' \
       "$HOST:tasochka/checkpoints/$RUN/" "$LOCAL/"; then
    echo "$(date +%H:%M) rsync failed (server down or network) — retry next round"
    return
  fi
  local step
  step=$(python3 -c "import json;print(json.load(open('$LOCAL/train_state.json'))['step'])" 2>/dev/null || echo 0)
  local dst
  dst=$(printf "%s/history/step_%06d" "$LOCAL" "$step")
  if [ "$step" -gt 0 ] && [ ! -d "$dst" ]; then
    mkdir -p "$dst"
    for f in weights.safetensors config.json tokenizer.json; do
      cp -c "$LOCAL/$f" "$dst/$f" 2>/dev/null || cp "$LOCAL/$f" "$dst/$f"
    done
    echo "$(date +%H:%M) step $step saved → $dst"
  else
    echo "$(date +%H:%M) synced, still step $step"
  fi
  # Нашли финальный шаг — дальше ждать нечего.
  if grep -q "saved final model" <(ssh -o BatchMode=yes "$HOST" "tail -3 tasochka/train.log" 2>/dev/null); then
    echo "$(date +%H:%M) training finished — final model is in $LOCAL"
    exit 0
  fi
}

if [ "$MODE" = "--once" ]; then
  sync_once
  exit 0
fi

echo "syncing $HOST:$RUN every ${MODE}s → $LOCAL (Ctrl+C to stop)"
# caffeinate -i: не давать Mac уснуть от бездействия, пока жив этот процесс.
caffeinate -i -w $$ &
while true; do
  sync_once
  sleep "$MODE"
done
