#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

python3 scripts/train.py \
  --device cuda \
  --output-dir runs/photon_ct_v4_phase1 \
  --epochs 50 \
  --steps-per-epoch 1000 \
  --val-steps 300 \
  --batch-size 2 \
  --gradient-accumulation 2 \
  --patch-size 96 96 96 \
  --inference-patch-size 128 128 128 \
  --base-channels 32 \
  --levels 5 \
  --blocks-per-level 2 \
  --dropout 0.1 \
  --physics-priors \
  --deep-supervision \
  --deep-supervision-weight 0.3 \
  --augment \
  --aug-hu-jitter 50.0 \
  --aug-density-scale 0.05 \
  --aug-noise-std 0.02 \
  --learning-rate 1e-4 \
  --weight-decay 2e-4 \
  --high-dose-weight 3.0 \
  --official-mae-weight 3.0 \
  --gradient-loss-weight 0.2 \
  --idd-surrogate-weight 0.5 \
  --scale-loss-weight 0.1 \
  --positive-patch-probability 0.85 \
  --num-workers 4 \
  --ct-cache-size 2 \
  --full-val-every 2 \
  --full-val-samples 15 \
  --full-val-records-per-patient 3 \
  --selection-metric full_mae \
  --inference-overlap 0.25 \
  --inference-batch-size 4
