#!/usr/bin/env bash
# v6.1: fine-tune of v6 targeting the two remaining measured error sources.
#   - --scale-head: bounded per-control-point global calibration (diagnosis
#     showed the optimal per-CP scale spans 0.80-1.13 and recovers ~9% MAE).
#   - --out-of-field-weight: penalize dose hallucinated where the reference
#     is essentially cold; the official masked MAE never sees those voxels
#     but the IDD curve integrates them (our worst metric vs the leaders).
# Warm-starts from the v6 best-full checkpoint; only scale-head weights are new.
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

INIT_CHECKPOINT=${1:-runs/photon_ct_v6_radiological/best_full.pt}

python3 scripts/train.py \
  --device cuda \
  --output-dir runs/photon_ct_v6_1_calibrated \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --epochs 40 \
  --steps-per-epoch 1000 \
  --val-steps 300 \
  --batch-size 4 \
  --gradient-accumulation 1 \
  --patch-size 96 96 96 \
  --inference-patch-size 128 128 128 \
  --base-channels 24 \
  --levels 5 \
  --blocks-per-level 2 \
  --dropout 0.1 \
  --physics-priors \
  --radiological-depth \
  --scale-head \
  --deep-supervision \
  --deep-supervision-weight 0.3 \
  --augment \
  --aug-hu-jitter 50.0 \
  --aug-density-scale 0.05 \
  --aug-noise-std 0.02 \
  --learning-rate 5e-5 \
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
