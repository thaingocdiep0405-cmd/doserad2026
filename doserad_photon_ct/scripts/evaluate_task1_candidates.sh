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

evaluate_candidate v2_reference \
  runs/photon_ct_v2_density/blend_full50_idd50.pt
evaluate_candidate v3_best_full \
  runs/photon_ct_v3_physics/best_full.pt
evaluate_candidate v3_best_idd \
  runs/photon_ct_v3_physics/best_idd.pt
evaluate_candidate v3_last \
  runs/photon_ct_v3_physics/last.pt
evaluate_candidate v3_blend_full25_idd75 \
  runs/photon_ct_v3_physics/blend_full25_idd75.pt
evaluate_candidate v3_blend_full50_idd50 \
  runs/photon_ct_v3_physics/blend_full50_idd50.pt
evaluate_candidate v3_blend_full75_idd25 \
  runs/photon_ct_v3_physics/blend_full75_idd25.pt

echo "Task 1 candidate evaluation complete: $OUTPUT_DIR"
