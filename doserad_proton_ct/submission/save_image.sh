#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TASK=${1:-proton-ct}
OUTPUT=${2:-"$PROJECT_DIR/dist/doserad2026-$TASK.tar.gz"}

mkdir -p "$(dirname -- "$OUTPUT")"
bash "$PROJECT_DIR/submission/build.sh" "$TASK"
docker image inspect "doserad2026-$TASK:latest" >/dev/null
docker save "doserad2026-$TASK:latest" | gzip -c > "$OUTPUT"
echo "Created $OUTPUT"
