#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

python3 scripts/train.py \
  --modality mri \
  --device cuda \
  --output-dir runs/proton_mri_v4_phase1 \
  --dose-scale 1e-4 \
  --epochs 50 \
  --steps-per-epoch 1000 \
  --val-steps 300 \
  --batch-size 2 \
  --gradient-accumulation 2 \
  --patch-size 96 96 96 \
  --base-channels 16 \
  --levels 4 \
  --blocks-per-level 2 \
  --dropout 0.1 \
  --deep-supervision \
  --deep-supervision-weight 0.3 \
  --augment \
  --aug-energy-jitter 0.02 \
  --aug-density-scale 0.03 \
  --aug-noise-std 0.02 \
  --learning-rate 1e-4 \
  --weight-decay 2e-4 \
  --high-dose-weight 3.0 \
  --official-mae-weight 3.0 \
  --gradient-loss-weight 0.2 \
  --idd-surrogate-weight 0.5 \
  --scale-loss-weight 0.1 \
  --positive-patch-probability 0.9 \
  --num-workers 4 \
  --cache-size 2
