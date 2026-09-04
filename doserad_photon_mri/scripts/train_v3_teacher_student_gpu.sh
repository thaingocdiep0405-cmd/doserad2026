#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python scripts/train.py \
  --manifest artifacts/manifest.csv \
  --splits artifacts/splits.json \
  --init-checkpoint runs/photon_mri_baseline/best_idd.pt \
  --output-dir runs/photon_mri_v3_teacher_student \
  --epochs 30 \
  --steps-per-epoch 1000 \
  --val-steps 150 \
  --batch-size 4 \
  --gradient-accumulation 1 \
  --num-workers 4 \
  --image-cache-size 4 \
  --ct-cache-size 4 \
  --patch-size 96 96 96 \
  --inference-patch-size 128 128 128 \
  --positive-patch-probability 0.85 \
  --physics-priors \
  --density-aux-weight 0.2 \
  --learning-rate 3e-5 \
  --weight-decay 2e-4 \
  --high-dose-weight 2.0 \
  --gradient-loss-weight 0.1 \
  --official-mae-weight 2.0 \
  --idd-surrogate-weight 0.15 \
  --scale-loss-weight 0.05 \
  --gradient-clip 1.0 \
  --full-val-every 2 \
  --full-val-samples 15 \
  --full-val-records-per-patient 3 \
  --selection-metric full_mae \
  --inference-overlap 0.25 \
  --inference-batch-size 4 \
  --device cuda \
  --log-every 20
