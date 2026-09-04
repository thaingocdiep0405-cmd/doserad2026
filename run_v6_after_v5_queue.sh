#!/usr/bin/env bash
# Waits for the v5 training queue to finish, then starts photon-CT v6.
# Safe to kill at any time; start with:
#   nohup bash run_v6_after_v5_queue.sh > v5_queue_logs/v6_watcher.out 2>&1 &
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG="$ROOT/v5_queue_logs/queue.log"
echo "[v6-watcher] $(date '+%F %T') waiting for v5 queue to complete"
while ! grep -q "ALL STAGES COMPLETE" "$LOG" 2>/dev/null; do
  sleep 300
done
echo "[v6-watcher] $(date '+%F %T') v5 queue complete, starting v6"
bash "$ROOT/doserad_photon_ct/scripts/train_v6_radiological.sh" \
  >> "$ROOT/v5_queue_logs/t1_photon_ct_v6.log" 2>&1
echo "[v6-watcher] $(date '+%F %T') v6 finished (exit $?)"
