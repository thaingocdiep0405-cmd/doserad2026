#!/usr/bin/env bash
# Waits for v6.1 to exit, then fine-tunes v6.2 (PB distill, patch 128).
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CKPT="$ROOT/doserad_photon_ct/runs/photon_ct_v6_1_calibrated/best_full.pt"
echo "[v6.2-watcher] $(date '+%F %T') waiting for v6.1 to finish"
while pgrep -f "photon_ct_v6_1_[c]alibrated" > /dev/null; do sleep 300; done
if [ ! -f "$CKPT" ]; then
  echo "[v6.2-watcher] ABORT: $CKPT missing"; exit 1
fi
echo "[v6.2-watcher] $(date '+%F %T') starting v6.2"
bash "$ROOT/doserad_photon_ct/scripts/train_v6_2_pb_distill.sh" \
  >> "$ROOT/v5_queue_logs/t1_photon_ct_v6_2.log" 2>&1
echo "[v6.2-watcher] $(date '+%F %T') v6.2 finished (exit $?)"
