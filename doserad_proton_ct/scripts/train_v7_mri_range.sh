#!/usr/bin/env bash
# v7 proton MRI: adds the water-equivalent depth and analytic Bragg-peak prior
# channels. A convolutional network cannot integrate density along the beam, so
# until now it had no way to know where a beamlet stops — which is why the
# 20/08 diagnostic found peaks displaced by 12-17 voxels, peak heights at half
# the reference, and dose predicted past the stopping point on low energies,
# and why v6's 2x better patch-level IDD surrogate moved full-volume IDD by
# only 6.7%. Warm-started from the v6 epoch-28 weights with the two new stem
# channels zero-initialised, so training starts exactly at v6 and improves from
# there.
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

INIT_CKPT=${1:-runs/proton_mri_v6_idd/init_12ch.pt}
if [[ ! -f "$INIT_CKPT" ]]; then
  echo "Missing init checkpoint $INIT_CKPT" >&2
  exit 1
fi

python3 scripts/train.py \
  --modality mri \
  --device cuda \
  --output-dir runs/proton_mri_v7_range \
  --init-checkpoint "$INIT_CKPT" \
  --range-channels \
  --dose-scale 1e-4 \
  --epochs 30 \
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
  --learning-rate 8e-5 \
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
  --snapshot-every 3
