#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CT_DIR="$WORKSPACE/doserad_photon_ct"
MRI_DIR="$WORKSPACE/doserad_photon_mri"

mkdir -p "$CT_DIR/runs/photon_ct_v3_physics"
mkdir -p "$MRI_DIR/runs/photon_mri_v3_teacher_student"

echo "[$(date --iso-8601=seconds)] Starting Task 1: photon dose on CT"
PYTHONUNBUFFERED=1 bash "$CT_DIR/scripts/train_v3_physics_gpu.sh" \
  2>&1 | tee "$CT_DIR/runs/photon_ct_v3_physics/train.log"

echo "[$(date --iso-8601=seconds)] Task 1 finished; starting Task 2: photon dose on MRI"
PYTHONUNBUFFERED=1 bash "$MRI_DIR/scripts/train_v3_teacher_student_gpu.sh" \
  2>&1 | tee "$MRI_DIR/runs/photon_mri_v3_teacher_student/train.log"

echo "[$(date --iso-8601=seconds)] Task 1 and Task 2 training completed"
