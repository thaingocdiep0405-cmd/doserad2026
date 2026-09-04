#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"

DOSE_SCALE=$(jq -r '.sample_max_median' artifacts/audit.json)

python3 scripts/train.py \
  --modality ct \
  --device cuda \
  --output-dir runs/proton_ct_v2_ray_gate \
  --init-checkpoint runs/proton_ct_v1_physics/best.pt \
  --epochs 12 \
  --steps-per-epoch 1000 \
  --val-steps 300 \
  --batch-size 4 \
  --patch-size 96 96 96 \
  --dose-scale "$DOSE_SCALE" \
  --learning-rate 3e-5 \
  --weight-decay 3e-4 \
  --positive-patch-probability 0.8 \
  --idd-surrogate-weight 0.5 \
  --scale-loss-weight 0.1 \
  --ray-gate-threshold 1e-6 \
  --early-stopping-patience 4 \
  --early-stopping-min-delta 1e-4 \
  --num-workers 4 \
  --cache-size 1

python3 scripts/train.py \
  --modality mri \
  --device cuda \
  --output-dir ../doserad_proton_mri/runs/proton_mri_v2_ray_gate \
  --init-checkpoint ../doserad_proton_mri/runs/proton_mri_v1_teacher_student/best.pt \
  --epochs 12 \
  --steps-per-epoch 1000 \
  --val-steps 300 \
  --batch-size 4 \
  --patch-size 96 96 96 \
  --dose-scale "$DOSE_SCALE" \
  --learning-rate 2e-5 \
  --weight-decay 3e-4 \
  --positive-patch-probability 0.8 \
  --idd-surrogate-weight 0.5 \
  --scale-loss-weight 0.1 \
  --ray-gate-threshold 1e-6 \
  --early-stopping-patience 4 \
  --early-stopping-min-delta 1e-4 \
  --num-workers 4 \
  --cache-size 1
