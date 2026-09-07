#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "creating venv with /usr/bin/python3"
  /usr/bin/python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "done. activate with:  source .venv/bin/activate"
echo "next steps:"
echo "  python -m tasochka.train               # train"
echo "  python -m tasochka.server              # start API"
echo "  open index.html                        # open chat UI"
