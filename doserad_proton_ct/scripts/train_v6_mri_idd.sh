#!/usr/bin/env bash
# v6 proton MRI, warm-started from v5. Targets the two IDD failure modes the
# 20/08 diagnostic found on validation beamlets:
#   * flattened Bragg peaks (predicted peak at 0.50x the reference height),
#     addressed by a heavier high-dose term and a 4x stronger IDD surrogate;
#   * dose hallucinated past the stopping point on low-energy beamlets
#     (tail error 0.19 at 46 MeV), addressed by the out-of-field term that v5
#     left switched off.
# Hidden feedback: MAE is only 2.3x off the top entries while IDD is 13x off,
# so shape, not magnitude, is what the remaining rank depends on.
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

INIT_CKPT=${1:-runs/proton_mri_v5/best.pt}
if [[ ! -f "$INIT_CKPT" ]]; then
  echo "Missing init checkpoint $INIT_CKPT" >&2
  exit 1
fi

python3 scripts/train.py \
  --modality mri \
  --device cuda \
  --output-dir runs/proton_mri_v6_idd \
  --init-checkpoint "$INIT_CKPT" \
  --dose-scale 1e-4 \
  --epochs 34 \
  --steps-per-epoch 1000 \
  --val-steps 300 \
  --batch-size 4 \
  --gradient-accumulation 1 \
  --patch-size 96 96 96 \
  --base-channels 16 \
  --levels 5 \
  --blocks-per-level 2 \
  --dropout 0.1 \
  --deep-supervision \
  --deep-supervision-weight 0.3 \
  --augment \
  --aug-energy-jitter 0.02 \
  --aug-density-scale 0.03 \
  --aug-noise-std 0.02 \
  --learning-rate 5e-5 \
  --weight-decay 2e-4 \
  --high-dose-weight 6.0 \
  --official-mae-weight 3.0 \
  --gradient-loss-weight 0.2 \
  --idd-surrogate-weight 2.0 \
  --scale-loss-weight 0.1 \
  --out-of-field-weight 0.5 \
  --positive-patch-probability 0.9 \
  --num-workers 6 \
  --cache-size 2 \
  --snapshot-every 4
