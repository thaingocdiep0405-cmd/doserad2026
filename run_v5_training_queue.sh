#!/usr/bin/env bash
# Sequential v5 training queue, ordered by leaderboard ROI:
#   1. T1 photon CT  (recipe validation + biggest field)
#   2. T4 proton MRI (12 entries, top-5 within reach)
#   3. T3 proton CT  (30 entries, already closest on accuracy)
#   4. T2 photon MRI (21 entries)
# Each stage logs separately; a failure does not block the rest of the queue.
set -u

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="$ROOT/v5_queue_logs"
mkdir -p "$LOG_DIR"

run_stage() {
  local name="$1" script="$2"
  echo "[queue] $(date '+%F %T') START $name" | tee -a "$LOG_DIR/queue.log"
  if bash "$script" >> "$LOG_DIR/$name.log" 2>&1; then
    echo "[queue] $(date '+%F %T') DONE  $name" | tee -a "$LOG_DIR/queue.log"
  else
    echo "[queue] $(date '+%F %T') FAIL  $name (exit $?)" | tee -a "$LOG_DIR/queue.log"
  fi
}

run_stage "t1_photon_ct_v5"  "$ROOT/doserad_photon_ct/scripts/train_v5_fast.sh"
run_stage "t4_proton_mri_v5" "$ROOT/doserad_proton_ct/scripts/train_v5_mri.sh"
run_stage "t3_proton_ct_v5"  "$ROOT/doserad_proton_ct/scripts/train_v5_ct.sh"
run_stage "t2_photon_mri_v5" "$ROOT/doserad_photon_mri/scripts/train_v5_fast.sh"

echo "[queue] $(date '+%F %T') ALL STAGES COMPLETE" | tee -a "$LOG_DIR/queue.log"
