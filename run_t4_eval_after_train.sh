#!/usr/bin/env bash
# Waits for the proton MRI v5 training to exit, then runs full-volume
# evaluation of both checkpoints in the configurations that matter for the
# submission decision. Results land in v5_queue_logs/t4_eval_*.log.
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJ="$ROOT/doserad_proton_ct"
LOG_DIR="$ROOT/v5_queue_logs"

echo "[t4-eval] $(date '+%F %T') waiting for proton MRI v5 training to finish"
while pgrep -f "train.py.*proton_mri_v5" > /dev/null; do
  sleep 300
done

CKPT_DIR="$PROJ/runs/proton_mri_v5"
if [[ ! -f "$CKPT_DIR/best.pt" ]]; then
  echo "[t4-eval] $(date '+%F %T') ABORT: $CKPT_DIR/best.pt not found"
  exit 1
fi

cd "$PROJ"
for ckpt in best last; do
  for overlap in 0.0 0.25; do
    tag="${ckpt}_ov${overlap/0./}"
    echo "[t4-eval] $(date '+%F %T') evaluating $ckpt.pt overlap=$overlap"
    python3 scripts/benchmark_roi_modes.py \
      --checkpoint "$CKPT_DIR/$ckpt.pt" \
      --modality mri \
      --max-patients 15 --records-per-patient 3 \
      --patch-size 96 96 96 --overlap "$overlap" \
      --modes capsule \
      --output "$LOG_DIR/t4_eval_${tag}.json" \
      > "$LOG_DIR/t4_eval_${tag}.log" 2>&1
  done
done
echo "[t4-eval] $(date '+%F %T') all evaluations complete"
grep -h "^\[capsule\]" "$LOG_DIR"/t4_eval_*.log 2>/dev/null | tail -8
