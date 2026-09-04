#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

python3 scripts/audit_dataset.py --check-headers --require-complete --expected-patients 75
python3 scripts/create_splits.py
python3 scripts/compute_sample_stats.py \
  --splits artifacts/splits.json \
  --split train \
  --max-patients 75 \
  --doses-per-patient 5
python3 scripts/check_readiness.py
