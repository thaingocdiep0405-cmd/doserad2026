#!/usr/bin/env bash
# Offline container smoke test for the Proton CT / MRI submission images.
#
# Mirrors the Grand Challenge lifecycle: start the container on an isolated
# network with no internet, wait for GET /health, mount the fixture read-only at
# /input, call POST /invoke, then validate the ten output slots.
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TASK=${1:-proton-ct}
BEAMLETS=${2:-4}
if [[ "$TASK" != "proton-ct" && "$TASK" != "proton-mri" ]]; then
  echo "Usage: $0 [proton-ct|proton-mri] [beamlets]" >&2
  exit 1
fi

IMAGE="doserad2026-$TASK:${IMAGE_TAG:-latest}"
# The pencil-beam backend ships a synthetic-CT checkpoint instead of a dose one,
# so which directory holds the model has to be selectable.
MODEL_DIR=${MODEL_DIR:-"$PROJECT_DIR/dist/$TASK"}
if ! compgen -G "$MODEL_DIR/*.pt" > /dev/null; then
  echo "No .pt checkpoint under $MODEL_DIR" >&2
  exit 1
fi

WORK=$(mktemp -d)
NETWORK="doserad-smoke-$$"
CONTAINER="doserad-smoke-$TASK-$$"
cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker network rm "$NETWORK" >/dev/null 2>&1 || true
  # Outputs belong to the container's non-root user, so remove them from inside
  # the image itself rather than fighting the permissions from the host.
  docker run --rm --user root --entrypoint sh \
    --volume "$WORK:/work" "$IMAGE" -c 'rm -rf /work/output' >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

python3 "$PROJECT_DIR/scripts/create_submission_smoke_fixture.py" \
  --output "$WORK/input" --task "$TASK" --beamlets "$BEAMLETS"
mkdir -p "$WORK/output"
chmod -R a+rX "$WORK/input"
chmod a+rwx "$WORK/output"

docker network create --internal "$NETWORK" >/dev/null
docker run --detach --name "$CONTAINER" \
  --network "$NETWORK" \
  --platform=linux/amd64 \
  --volume "$WORK/input:/input:ro" \
  --volume "$WORK/output:/output" \
  --volume "$MODEL_DIR:/opt/ml/model:ro" \
  --env BEAMLET_CHUNK_SIZE="${BEAMLET_CHUNK_SIZE:-2}" \
  --env INFERENCE_PATCH_SIZE=16,16,16 \
  "$IMAGE" >/dev/null

echo "Waiting for /health on $CONTAINER"
health=""
for attempt in $(seq 1 120); do
  health=$(docker exec "$CONTAINER" python -c "
import urllib.request
try:
    print(urllib.request.urlopen('http://localhost:4743/health', timeout=5).status)
except Exception:
    print('')
" 2>/dev/null | tr -d '\r')
  if [[ "$health" == "200" ]]; then
    echo "health=200 after ${attempt} attempt(s)"
    break
  fi
  sleep 5
done
if [[ "$health" != "200" ]]; then
  echo "FAIL: /health never returned 200" >&2
  docker logs "$CONTAINER" >&2 || true
  exit 1
fi

invoke=$(docker exec "$CONTAINER" python -c "
import urllib.request
request = urllib.request.Request('http://localhost:4743/invoke', method='POST')
print(urllib.request.urlopen(request, timeout=3600).status)
" | tr -d '\r')
echo "invoke=$invoke"
if [[ "$invoke" != "201" ]]; then
  echo "FAIL: /invoke returned $invoke" >&2
  docker logs "$CONTAINER" >&2
  exit 1
fi

docker logs "$CONTAINER" 2>&1 | tail -20
python3 "$PROJECT_DIR/submission/validate_outputs.py" "$WORK/output"

frames=$(python3 - "$WORK/output" "$BEAMLETS" <<'PY'
import sys
from pathlib import Path
import SimpleITK as sitk

root, expected = Path(sys.argv[1]), int(sys.argv[2])
path = next((root / "images" / "stacked-radiation-dose-map-1").glob("*.mha"))
image = sitk.ReadImage(str(path))
size = image.GetSize()
assert len(size) == 4, f"slot 1 is not 4D: {size}"
assert size[3] == expected, f"slot 1 holds {size[3]} frames, expected {expected}"
print(size[3])
PY
)
echo "PASS: $TASK offline smoke test, slot 1 stacked $frames frame(s)"
