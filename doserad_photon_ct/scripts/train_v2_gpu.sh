#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

python3 scripts/train.py \
  --device cuda \
  --output-dir runs/photon_ct_v2_density \
  --init-checkpoint runs/photon_ct_baseline/last.pt \
  --epochs 30 \
  --steps-per-epoch 1000 \
  --val-steps 200 \
  --batch-size 4 \
  --gradient-accumulation 1 \
  --patch-size 96 96 96 \
  --inference-patch-size 128 128 128 \
  --base-channels 12 \
  --levels 4 \
  --blocks-per-level 2 \
  --include-density \
  --learning-rate 5e-5 \
  --weight-decay 1e-4 \
  --high-dose-weight 4.0 \
  --gradient-loss-weight 0.1 \
  --positive-patch-probability 0.8 \
  --num-workers 4 \
  --ct-cache-size 1 \
  --full-val-every 2 \
  --full-val-samples 15 \
  --selection-metric full_mae \
  --inference-overlap 0.25 \
  --inference-batch-size 4
