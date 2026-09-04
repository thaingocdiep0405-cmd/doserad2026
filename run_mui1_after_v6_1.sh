#!/usr/bin/env bash
# Waits for photon v6.1 training to exit, then:
#   1. Task 1: full-volume evaluation of v6 vs v6.1 checkpoints (checkpoint pick)
#   2. Task 4: uncontended runtime sweep for the Mũi-1 resubmission config
# Logs land in v5_queue_logs/.
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="$ROOT/v5_queue_logs"

echo "[mui1] $(date '+%F %T') waiting for photon v6.1 training to finish"
while pgrep -f "[t]rain.py.*photon_ct_v6_1_calibrated" > /dev/null; do
  sleep 300
done
echo "[mui1] $(date '+%F %T') v6.1 finished"

cd "$ROOT/doserad_photon_ct"
for ckpt in runs/photon_ct_v6_radiological/best_full.pt \
            runs/photon_ct_v6_1_calibrated/best_full.pt; do
  tag=$(basename "$(dirname "$ckpt")")
  echo "[mui1] $(date '+%F %T') T1 eval $tag"
  python3 scripts/evaluate_checkpoint.py \
    --checkpoint "$ckpt" \
    --overlap 0.25 --batch-size 4 \
    --skip-empty-aperture --mask-outside-body --pad-batch \
    --output "$LOG_DIR/t1_eval_${tag}.json" \
    > "$LOG_DIR/t1_eval_${tag}.log" 2>&1
done

cd "$ROOT/doserad_proton_ct"
echo "[mui1] $(date '+%F %T') T4 runtime sweep"
python3 scripts/benchmark_t4_runtime_sweep.py \
  --checkpoint runs/proton_mri_v5/best.pt \
  --modality mri \
  --output "$LOG_DIR/t4_runtime_sweep.json" \
  > "$LOG_DIR/t4_runtime_sweep.log" 2>&1
echo "[mui1] $(date '+%F %T') all done"
