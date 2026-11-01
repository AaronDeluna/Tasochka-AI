#!/usr/bin/env bash
# Первичная настройка на Mac: .venv + MLX. На сервере см. docs/SERVER.md.
set -euo pipefail

cd "$(dirname "$0")"

# Берём самый свежий Python из доступных: новые версии MLX требуют 3.10+,
# системный /usr/bin/python3 (3.9) тоже работает, но застревает на mlx 0.29.
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 /usr/bin/python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$(command -v "$candidate")"
    break
  fi
done
[ -n "$PYTHON" ] || { echo "python3 не найден"; exit 1; }

# venv, скопированный с другой машины/пользователя, не запускается — пересоздаём.
if [ -d .venv ] && ! .venv/bin/python -c "" >/dev/null 2>&1; then
  echo "removing broken .venv"
  rm -rf .venv
fi

if [ ! -d .venv ]; then
  echo "creating venv with $PYTHON ($("$PYTHON" --version))"
  "$PYTHON" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/base.txt

echo
echo "done. activate with:  source .venv/bin/activate"
echo "next steps (запускать из корня проекта):"
echo "  python scripts/download_en.py --preset mac        # ~0.5 GB английских данных"
echo "  python -m tasochka.train --config configs/mac-smoke.json"
echo "  python -m tasochka.generate --checkpoint checkpoints/mac-smoke --chat"
