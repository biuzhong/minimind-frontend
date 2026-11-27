#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_BIN="${PY_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
LOG_DIR="${LOG_DIR:-$ROOT/runs/stages}"

case "${1:-student}" in
  student)
    parent=expand_pre
    width=384
    layers=6
    epochs=6
    batch=32
    lr=0.0003
    tag=expand_sft
    data="$ROOT/experiments/sft/data/expanded/sft.jsonl"
    ;;
  teacher)
    cd "$ROOT"
    "$PY_BIN" - <<'PYTHON'
from pathlib import Path
from shutil import copyfile
from experiments.run import fetch, sha256

source = fetch("full_sft_768.pth", Path(".assets"))
target = Path("out/expand_teacher_base_768.pth")
target.parent.mkdir(exist_ok=True)
if not target.exists():
    copyfile(source, target)
elif sha256(target) != sha256(source):
    raise ValueError(f"Invalid cached teacher weights: {target}")
PYTHON
    parent=expand_teacher_base
    width=768
    layers=8
    epochs=3
    batch=24
    lr=0.00015
    tag=expand_teacher
    data="$ROOT/experiments/distillation/data/expanded/teacher_sft.jsonl"
    ;;
  *) printf 'Usage: %s [student|teacher]\n' "$0" >&2; exit 2 ;;
esac

if [[ ! -f "$ROOT/out/${parent}_${width}.pth" ]]; then
  printf 'Missing input checkpoint: %s\n' "$ROOT/out/${parent}_${width}.pth" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
cd "$ROOT/trainer"
"$PY_BIN" -u ../experiments/sft/train_full_sft.py \
  --device "$DEVICE" --epochs "$epochs" --batch_size "$batch" --num_workers 0 \
  --accumulation_steps 1 --max_seq_len 128 --save_interval 1000 --log_interval 20 \
  --data_path "$data" --save_weight "$tag" --hidden_size "$width" --num_hidden_layers "$layers" \
  --learning_rate "$lr" --from_weight "$parent" 2>&1 | tee "$LOG_DIR/$tag.log"
