#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

mkdir -p runs ../doserad_proton_mri/runs
printf '%s\n' "$$" > runs/train_all.pid
DOSE_SCALE=$(jq -r '.sample_max_median' artifacts/audit.json)

python3 scripts/train.py \
  --modality ct --device cuda \
  --output-dir runs/proton_ct_v1_physics \
  --epochs 30 --steps-per-epoch 1000 --val-steps 300 \
  --batch-size 4 --patch-size 96 96 96 --dose-scale "$DOSE_SCALE" \
  --num-workers 4 --cache-size 1

python3 scripts/train.py \
  --modality mri --device cuda \
  --output-dir ../doserad_proton_mri/runs/proton_mri_v1_teacher_student \
  --init-checkpoint runs/proton_ct_v1_physics/best.pt \
  --epochs 30 --steps-per-epoch 1000 --val-steps 300 \
  --batch-size 4 --patch-size 96 96 96 --dose-scale "$DOSE_SCALE" \
  --learning-rate 5e-5 --num-workers 4 --cache-size 1
