#!/usr/bin/env bash
# Waits for the running photon v6 training to exit, then trains the proton v5
# chain: CT v5 first, then MRI v5 warm-started from the CT v5 checkpoint.
# Note: the v6.1 watcher fine-tunes photon v6.1 on the same GPU concurrently;
# both jobs fit in memory and simply share throughput until v6.1 completes.
# Safe to kill at any time. Start with:
#   nohup bash run_proton_v5_after_v6.sh > v5_queue_logs/proton_v5_watcher.out 2>&1 &
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="$ROOT/v5_queue_logs"

echo "[proton-v5] $(date '+%F %T') waiting for photon v6 training to finish"
while pgrep -f "train.py.*photon_ct_v6_radiological" > /dev/null; do
  sleep 300
done

echo "[proton-v5] $(date '+%F %T') v6 finished, starting proton CT v5"
bash "$ROOT/doserad_proton_ct/scripts/train_v5_ct.sh" \
  >> "$LOG_DIR/t3_proton_ct_v5.log" 2>&1
CT_EXIT=$?
echo "[proton-v5] $(date '+%F %T') proton CT v5 finished (exit $CT_EXIT)"

CKPT="$ROOT/doserad_proton_ct/runs/proton_ct_v5/best.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "[proton-v5] $(date '+%F %T') ABORT: $CKPT not found (CT v5 crashed?)"
  exit 1
fi

echo "[proton-v5] $(date '+%F %T') starting proton MRI v5 (warm from CT v5)"
bash "$ROOT/doserad_proton_ct/scripts/train_v5_mri_warm.sh" "$CKPT" \
  >> "$LOG_DIR/t4_proton_mri_v5.log" 2>&1
echo "[proton-v5] $(date '+%F %T') proton MRI v5 finished (exit $?)"
