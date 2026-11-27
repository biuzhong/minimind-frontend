#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_BIN="${PY_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
LOG_DIR="${LOG_DIR:-$ROOT/runs/stages}"

case "${1:-kd}" in
  kd)
    alpha=0.5
    tag=expand_kd
    if [[ ! -f "$ROOT/out/expand_teacher_768.pth" ]]; then
      printf 'Run the teacher SFT stage to produce out/expand_teacher_768.pth.\n' >&2
      exit 1
    fi
    ;;
  ce) alpha=1; tag=expand_ce ;;
  *) printf 'Usage: %s [kd|ce]\n' "$0" >&2; exit 2 ;;
esac

if [[ ! -f "$ROOT/out/expand_rl_384.pth" ]]; then
  printf 'Run the RL stage to produce out/expand_rl_384.pth.\n' >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
cd "$ROOT/trainer"
"$PY_BIN" -u ../experiments/distillation/train_distillation.py \
  --device "$DEVICE" --epochs 1 --batch_size 16 --num_workers 0 \
  --accumulation_steps 1 --max_seq_len 128 --save_interval 1000 --log_interval 20 \
  --data_path "$ROOT/experiments/distillation/data/expanded/teacher_sft.jsonl" \
  --save_weight "$tag" --learning_rate 5e-05 --teacher_use_moe 0 \
  --teacher_hidden_size 768 --teacher_num_layers 8 \
  --from_student_weight expand_rl --from_teacher_weight expand_teacher \
  --alpha "$alpha" --temperature 2 --student_hidden_size 384 --student_num_layers 6 \
  2>&1 | tee "$LOG_DIR/$tag.log"
