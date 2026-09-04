#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 /path/to/best.pt /path/to/model.tar.gz"
  exit 1
fi

CHECKPOINT=$(realpath "$1")
OUTPUT=$(realpath -m "$2")
mkdir -p "$(dirname -- "$OUTPUT")"
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

cp "$CHECKPOINT" "$TEMP_DIR/best.pt"
tar -C "$TEMP_DIR" -czf "$OUTPUT" best.pt
echo "Created $OUTPUT"
