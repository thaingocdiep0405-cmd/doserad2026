#!/usr/bin/env bash
# Waits for the v7 training to exit, then evaluates the final candidates on the
# full-volume protocol. Patch validation has twice picked checkpoints that did
# not hold up on full volumes, so every candidate is measured before one is
# chosen for the submission.
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJ="$ROOT/doserad_proton_ct"
OUT="$ROOT/v5_queue_logs"

echo "[v7-eval] $(date '+%F %T') waiting for v7 training to finish"
while pgrep -f "[t]rain.py.*proton_mri_v7_range" > /dev/null; do
  sleep 120
done
echo "[v7-eval] $(date '+%F %T') training finished"

cd "$PROJ"
for ckpt in best last epoch030 epoch027; do
  [ -f "runs/proton_mri_v7_range/$ckpt.pt" ] || continue
  echo "[v7-eval] $(date '+%F %T') evaluating $ckpt"
  python3 scripts/benchmark_roi_modes.py \
    --checkpoint "runs/proton_mri_v7_range/$ckpt.pt" \
    --modality mri --max-patients 15 --records-per-patient 3 \
    --patch-size 96 96 96 --modes bbox \
    --output "$OUT/v7_final_${ckpt}.json" \
    > "$OUT/v7_final_${ckpt}.log" 2>&1
  grep "^\[bbox\]" "$OUT/v7_final_${ckpt}.log" | sed "s/^/[$ckpt] /"
done
echo "[v7-eval] $(date '+%F %T') all evaluations complete"
