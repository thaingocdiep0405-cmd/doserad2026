#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
docker build \
  --platform=linux/amd64 \
  --file "$PROJECT_DIR/submission/Dockerfile" \
  --tag doserad2026-photon-mri:latest \
  "$PROJECT_DIR"
