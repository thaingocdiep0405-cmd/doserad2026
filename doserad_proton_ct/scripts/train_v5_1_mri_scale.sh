#!/usr/bin/env bash
# T4 proton-MRI v5.1: fine-tune of proton_mri_v5 with the bounded per-beamlet
# scale head ported from the photon project. Motivation: full-volume eval of
# v5 shows scale_ratio 0.97 +/- 0.10 across beamlets — exactly the error class
# the head removes. Keep patch 96 (measured better than 128 on BOTH MAE and
# IDD for proton) and raise the scale-loss weight.
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

python3 scripts/train.py \
  --modality mri \
  --output-dir runs/proton_mri_v5_1_scale \
  --init-checkpoint runs/proton_mri_v5/best.pt \
  --epochs 20 \
  --steps-per-epoch 1000 \
  --val-steps 300 \
  --batch-size 4 \
  --patch-size 96 96 96 \
  --base-channels 16 \
  --levels 5 \
  --blocks-per-level 2 \
  --dropout 0.1 \
  --scale-head \
  --deep-supervision \
  --deep-supervision-weight 0.3 \
  --augment \
  --learning-rate 3e-5 \
  --weight-decay 2e-4 \
  --high-dose-weight 3.0 \
  --official-mae-weight 3.0 \
  --gradient-loss-weight 0.2 \
  --idd-surrogate-weight 0.5 \
  --scale-loss-weight 0.3 \
  --out-of-field-weight 0.5 \
  --positive-patch-probability 0.85 \
  --num-workers 6 \
  --cache-size 2 \
  --early-stopping-patience 8 \
  --seed 2026
