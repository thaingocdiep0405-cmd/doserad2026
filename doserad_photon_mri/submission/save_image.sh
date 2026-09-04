#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${1:-"$PROJECT_DIR/dist/doserad2026-photon-mri.tar.gz"}

mkdir -p "$(dirname -- "$OUTPUT")"
bash "$PROJECT_DIR/submission/build.sh"
docker image inspect doserad2026-photon-mri:latest >/dev/null
docker save doserad2026-photon-mri:latest | gzip -c > "$OUTPUT"
echo "Created $OUTPUT"
