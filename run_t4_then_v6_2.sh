#!/usr/bin/env bash
# After photon v6.1: run T4 proton-MRI v5.1 first (user priority), then photon v6.2.
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
echo "[t4-chain] $(date '+%F %T') waiting for photon v6.1 to finish"
while pgrep -f "photon_ct_v6_1_[c]alibrated" > /dev/null; do sleep 300; done

echo "[t4-chain] $(date '+%F %T') starting T4 proton MRI v5.1"
bash "$ROOT/doserad_proton_ct/scripts/train_v5_1_mri_scale.sh" \
  >> "$ROOT/v5_queue_logs/t4_proton_mri_v5_1.log" 2>&1
echo "[t4-chain] $(date '+%F %T') T4 v5.1 finished (exit $?)"

CKPT="$ROOT/doserad_photon_ct/runs/photon_ct_v6_1_calibrated/best_full.pt"
if [ -f "$CKPT" ]; then
  echo "[t4-chain] $(date '+%F %T') starting photon v6.2"
  bash "$ROOT/doserad_photon_ct/scripts/train_v6_2_pb_distill.sh" \
    >> "$ROOT/v5_queue_logs/t1_photon_ct_v6_2.log" 2>&1
  echo "[t4-chain] $(date '+%F %T') v6.2 finished (exit $?)"
else
  echo "[t4-chain] SKIP v6.2: $CKPT missing"
fi
