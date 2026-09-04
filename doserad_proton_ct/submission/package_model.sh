#!/usr/bin/env bash
# Build the model.tar.gz that Grand Challenge extracts to /opt/ml/model.
#
# The network backend needs one dose checkpoint, stored as best.pt. The
# pencil-beam backend needs no dose checkpoint at all but does need a
# synthetic-CT one, which it looks up as sct*.pt; pass "-" in place of the
# checkpoint for that case.
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 {/path/to/best.pt|-} /path/to/model.tar.gz [/path/to/sct.pt]" >&2
  exit 1
fi

CHECKPOINT=$1
OUTPUT=$(realpath -m "$2")
SYNTHETIC=${3:-}

if [ "$CHECKPOINT" = "-" ] && [ -z "$SYNTHETIC" ]; then
  echo "Nothing to package: no checkpoint and no synthetic CT" >&2
  exit 1
fi

mkdir -p "$(dirname -- "$OUTPUT")"
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

MEMBERS=()
if [ "$CHECKPOINT" != "-" ]; then
  cp "$(realpath "$CHECKPOINT")" "$TEMP_DIR/best.pt"
  MEMBERS+=("best.pt")
fi
if [ -n "$SYNTHETIC" ]; then
  cp "$(realpath "$SYNTHETIC")" "$TEMP_DIR/sct.pt"
  MEMBERS+=("sct.pt")
fi

tar -C "$TEMP_DIR" -czf "$OUTPUT" "${MEMBERS[@]}"
echo "Created $OUTPUT with: ${MEMBERS[*]}"
