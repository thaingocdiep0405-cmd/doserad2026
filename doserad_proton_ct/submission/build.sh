#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=$(cd -- "$PROJECT_DIR/.." && pwd)
TASK=${1:-proton-ct}
if [[ "$TASK" != "proton-ct" && "$TASK" != "proton-mri" ]]; then
  echo "Usage: $0 [proton-ct|proton-mri]"
  exit 1
fi

docker build \
  --platform=linux/amd64 \
  --build-arg "TASK=$TASK" \
  --file "$PROJECT_DIR/submission/Dockerfile" \
  --tag "doserad2026-$TASK:latest" \
  "$WORKSPACE"
