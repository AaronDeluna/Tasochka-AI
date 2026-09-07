#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv server/.venv
server/.venv/bin/python -m pip install --upgrade pip
server/.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
server/.venv/bin/python -m pip install -r server/requirements-distributed.txt
printf '%s\n' 'Activate: source server/.venv/bin/activate' 'Read server/README.md before a full training run.'
