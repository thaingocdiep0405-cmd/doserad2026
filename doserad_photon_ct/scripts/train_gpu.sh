#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

python3 scripts/train.py \
  --device cuda \
  --output-dir runs/photon_ct_baseline \
  --epochs 100 \
  --steps-per-epoch 1000 \
  --val-steps 100 \
  --batch-size 4 \
  --gradient-accumulation 1 \
  --patch-size 96 96 96 \
  --base-channels 12 \
  --levels 4 \
  --blocks-per-level 2 \
  --num-workers 4 \
  --ct-cache-size 1 \
  --full-val-every 5 \
  --full-val-samples 15 \
  --selection-metric full_mae \
  --inference-overlap 0.25 \
  --inference-batch-size 4
