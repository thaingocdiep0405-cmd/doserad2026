#!/usr/bin/env bash
# v6.2: v6.1 recipe + pencil-beam distillation.
# An auxiliary head predicts the precomputed pyRadPlan SVDPB dose from the
# shared features. PB alone measures masked MAE ~0.045 (better than our v2
# network, far from MC truth), so it feeds a separate head as a physics
# teacher instead of pulling the main MC-supervised output toward PB errors.
# Requires artifacts/pb_doses (scripts/precompute_pb_doses.py).
# Warm-starts from the v6.1 best checkpoint; pb-head weights are new.
# Patch 128 matches the inference patch: measured on v6.1, inference at the
# training size (96) gains 5% MAE but doubles IDD, so training moves to 128
# instead to align the two without giving up the large-context IDD win.
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

INIT_CHECKPOINT=${1:-runs/photon_ct_v6_1_calibrated/best_full.pt}

python3 scripts/train.py \
  --device cuda \
  --output-dir runs/photon_ct_v6_2_pb_distill \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --epochs 30 \
  --steps-per-epoch 1000 \
  --val-steps 300 \
  --batch-size 2 \
  --gradient-accumulation 2 \
  --patch-size 128 128 128 \
  --inference-patch-size 128 128 128 \
  --base-channels 24 \
  --levels 5 \
  --blocks-per-level 2 \
  --dropout 0.1 \
  --physics-priors \
  --radiological-depth \
  --scale-head \
  --pb-dose-dir artifacts/pb_doses \
  --pb-distill-weight 0.3 \
  --deep-supervision \
  --deep-supervision-weight 0.3 \
  --augment \
  --aug-hu-jitter 50.0 \
  --aug-density-scale 0.05 \
  --aug-noise-std 0.02 \
  --learning-rate 3e-5 \
  --weight-decay 2e-4 \
  --high-dose-weight 3.0 \
  --official-mae-weight 3.0 \
  --gradient-loss-weight 0.2 \
  --idd-surrogate-weight 0.5 \
  --scale-loss-weight 0.1 \
  --out-of-field-weight 0.5 \
  --out-of-field-threshold 0.02 \
  --positive-patch-probability 0.85 \
  --num-workers 6 \
  --ct-cache-size 2 \
  --full-val-every 2 \
  --full-val-samples 15 \
  --full-val-records-per-patient 3 \
  --selection-metric full_mae \
  --inference-overlap 0.25 \
  --inference-batch-size 4
