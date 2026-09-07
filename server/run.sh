#!/usr/bin/env bash
# Run from any directory. All persistent training paths are inside server/.
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
GPUS="${GPUS:-8}"
case "${1:-}" in
  data)
    python -m server.download
    if [[ ! -f server/data/tokenizer/tokenizer.json ]]; then
      python -m server.tokenizer
    fi
    python -m server.prepare
    python -m server.preflight
    ;;
  pretrain|16k|32k|128k|sft)
    stage="$1"
    shift
    args=(--output "server/runs/$stage" --deepspeed server/configs/zero3.json)
    case "$stage" in
      pretrain) args+=(--context 4096 --tokens 60000000000 --accumulation 32 --lr 0.0003) ;;
      16k) args+=(--mode long --init-from server/runs/pretrain/final --sources long_books --context 16384 --tokens 1000000000 --accumulation 8 --lr 0.00003) ;;
      32k) args+=(--mode long --init-from server/runs/16k/final --sources long_books --context 32768 --tokens 1000000000 --accumulation 4 --lr 0.00002) ;;
      128k) args+=(--mode long --init-from server/runs/32k/final --sources long_books --context 131072 --tokens 1000000000 --accumulation 1 --lr 0.00001 --eval-limit 8) ;;
      sft) args+=(--mode sft --init-from server/runs/128k/final --context 4096 --max-steps 1000 --accumulation 4 --lr 0.00001) ;;
    esac
    torchrun --standalone --nproc_per_node="$GPUS" -m server.train "${args[@]}" "$@"
    ;;
  all)
    bash server/run.sh data
    for stage in pretrain 16k 32k 128k sft; do
      if [[ -f "server/runs/$stage/final/stage_complete.json" ]]; then
        printf 'Completed: %s\n' "$stage"
      elif [[ -f "server/runs/$stage/run.json" ]]; then
        bash server/run.sh "$stage" --resume
      else
        bash server/run.sh "$stage"
      fi
    done
    ;;
  *) printf '%s\n' 'Usage: bash server/run.sh {data|pretrain|16k|32k|128k|sft|all} [train arguments]' >&2; exit 2 ;;
esac
