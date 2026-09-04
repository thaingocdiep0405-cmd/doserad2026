#!/usr/bin/env bash
# Waits for the running v6 training to exit, then fine-tunes v6.1 from its
# best-full checkpoint. Safe to kill at any time. Start with:
#   nohup bash run_v6_1_after_v6.sh > v5_queue_logs/v6_1_watcher.out 2>&1 &
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CKPT="$ROOT/doserad_photon_ct/runs/photon_ct_v6_radiological/best_full.pt"

echo "[v6.1-watcher] $(date '+%F %T') waiting for v6 training to finish"
while pgrep -f "train.py.*photon_ct_v6_radiological" > /dev/null; do
  sleep 300
done
if [ ! -f "$CKPT" ]; then
  echo "[v6.1-watcher] $(date '+%F %T') ABORT: $CKPT not found (v6 crashed early?)"
  exit 1
fi
echo "[v6.1-watcher] $(date '+%F %T') v6 finished, starting v6.1 fine-tune"
bash "$ROOT/doserad_photon_ct/scripts/train_v6_1_calibrated.sh" \
  >> "$ROOT/v5_queue_logs/t1_photon_ct_v6_1.log" 2>&1
echo "[v6.1-watcher] $(date '+%F %T') v6.1 finished (exit $?)"
