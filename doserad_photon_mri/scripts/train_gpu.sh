#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

python3 scripts/train.py \
  --device cuda \
  --output-dir runs/photon_mri_baseline \
  --init-checkpoint "$PROJECT_DIR/../doserad_photon_ct/runs/photon_ct_baseline/last.pt" \
  --epochs 40 \
  --steps-per-epoch 1000 \
  --val-steps 200 \
  --batch-size 4 \
  --gradient-accumulation 1 \
  --patch-size 96 96 96 \
  --base-channels 12 \
  --levels 4 \
  --blocks-per-level 2 \
  --learning-rate 0.00005 \
  --num-workers 4 \
  --image-cache-size 1 \
  --full-val-every 2 \
  --full-val-samples 15 \
  --selection-metric full_mae \
  --inference-overlap 0.25 \
  --inference-batch-size 4 \
  --inference-patch-size 128 128 128
