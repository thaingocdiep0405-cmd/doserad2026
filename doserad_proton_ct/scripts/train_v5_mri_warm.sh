#!/usr/bin/env bash
# v5 proton MRI, warm-started from the proton CT v5 checkpoint. The ten-channel
# input contract is identical across CT/MRI, and the v1 MRI model that
# warm-started from CT matched CT accuracy despite MRI carrying no density —
# training MRI from scratch forfeits that transfer.
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

INIT_CKPT=${1:-runs/proton_ct_v5/best.pt}
if [[ ! -f "$INIT_CKPT" ]]; then
  echo "Missing init checkpoint $INIT_CKPT" >&2
  exit 1
fi

python3 scripts/train.py \
  --modality mri \
  --device cuda \
  --output-dir runs/proton_mri_v5 \
  --init-checkpoint "$INIT_CKPT" \
  --dose-scale 1e-4 \
  --epochs 50 \
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
  --learning-rate 1e-4 \
  --weight-decay 2e-4 \
  --high-dose-weight 3.0 \
  --official-mae-weight 3.0 \
  --gradient-loss-weight 0.2 \
  --idd-surrogate-weight 0.5 \
  --scale-loss-weight 0.1 \
  --positive-patch-probability 0.9 \
  --num-workers 6 \
  --cache-size 2
