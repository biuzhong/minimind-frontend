#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
base=/root/autodl-tmp/minimind-1b
cd "$base/pinned-source"
echo GitHub commit 0a94dc57941abca9dd6b05ca6121b844efaf974e
until [[ -s "$base/teacher-base/model-00000-of-00001.safetensors" ]]; do sleep 12; done
echo "7ab8fd86563125929be78aeec8cb3969c7ed2ead3be1ab9d3ec0a9fa69c8660d  $base/teacher-base/model-00000-of-00001.safetensors" | sha256sum -c -
date -Is
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -u experiments/distill_1b.py --run-dir ../../minimind-expanded/records --teacher-dir "$base/teacher-base" --output "$base/records"
date -Is
