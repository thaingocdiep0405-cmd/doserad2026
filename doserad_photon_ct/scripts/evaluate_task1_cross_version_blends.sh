#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

OUTPUT_DIR="artifacts/task1_candidate_evaluations"
mkdir -p "$OUTPUT_DIR"

evaluate_candidate() {
  local name="$1"
  local checkpoint="$2"
  echo "Evaluating $name: $checkpoint"
  python scripts/evaluate_checkpoint.py \
    --checkpoint "$checkpoint" \
    --max-records 75 \
    --patch-size 128 128 128 \
    --overlap 0.25 \
    --batch-size 4 \
    --bootstrap-samples 2000 \
    --confidence-level 0.95 \
    --skip-empty-aperture \
    --mask-outside-body \
    --pad-batch \
    --device cuda \
    --output "$OUTPUT_DIR/$name.json"
}

# The v2 checkpoint is zero-padded from 7 to 11 input channels before blending.
# This keeps the Task 1 inference architecture unchanged and adds no runtime cost.
evaluate_candidate blend_v2_10_last90 \
  runs/photon_ct_v3_physics/blend_v2_10_last90.pt
evaluate_candidate blend_v2_25_last75 \
  runs/photon_ct_v3_physics/blend_v2_25_last75.pt
evaluate_candidate blend_v2_10_bestidd90 \
  runs/photon_ct_v3_physics/blend_v2_10_bestidd90.pt
evaluate_candidate blend_v2_25_bestidd75 \
  runs/photon_ct_v3_physics/blend_v2_25_bestidd75.pt

echo "Task 1 cross-version blend evaluation complete: $OUTPUT_DIR"
