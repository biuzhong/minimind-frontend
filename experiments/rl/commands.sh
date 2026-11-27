#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_BIN="${PY_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
LOG_DIR="${LOG_DIR:-$ROOT/runs/stages}"

if [[ ! -f "$ROOT/out/expand_sft_384.pth" ]]; then
  printf 'Run the SFT student stage to produce out/expand_sft_384.pth.\n' >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
cd "$ROOT/trainer"
"$PY_BIN" -u ../experiments/rl/train_grpo.py \
  --device "$DEVICE" --epochs 3 --batch_size 8 --num_workers 0 \
  --accumulation_steps 1 --max_seq_len 128 --save_interval 1000 --log_interval 8 \
  --data_path "$ROOT/experiments/rl/data/expanded/rl.jsonl" --save_weight expand_rl \
  --hidden_size 384 --num_hidden_layers 6 --learning_rate 7e-05 \
  --max_gen_len 30 --num_generations 6 --thinking_ratio 0 --beta 0.02 \
  --loss_type grpo --http_demo_rewards --rollout_temperature 1.05 --debug_mode \
  --debug_interval 40 --from_weight expand_sft 2>&1 | tee "$LOG_DIR/rl.log"
