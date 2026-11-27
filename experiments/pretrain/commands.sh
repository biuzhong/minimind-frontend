#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_BIN="${PY_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
LOG_DIR="${LOG_DIR:-$ROOT/runs/stages}"

mkdir -p "$LOG_DIR"
cd "$ROOT"
"$PY_BIN" - <<'PYTHON'
from scripts.sharing_expanded import STUDENT, save_random, weights

initial = weights("expand_random", STUDENT)
if not initial.exists():
    save_random(initial)
PYTHON

cd "$ROOT/trainer"
"$PY_BIN" -u ../experiments/pretrain/train_pretrain.py \
  --device "$DEVICE" --epochs 2 --batch_size 32 --num_workers 0 \
  --accumulation_steps 1 --max_seq_len 128 --save_interval 1000 --log_interval 20 \
  --data_path "$ROOT/experiments/pretrain/data/expanded/pretrain.jsonl" \
  --save_weight expand_pre --hidden_size 384 --num_hidden_layers 6 \
  --learning_rate 0.0007 --from_weight expand_random 2>&1 | tee "$LOG_DIR/pretrain.log"
