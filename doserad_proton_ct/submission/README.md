# Proton CT/MRI submission adapter

This adapter implements the DoseRAD2026 Grand Challenge `/health` and
`/invoke` contract for Task 3 Proton CT and Task 4 Proton MRI. One shared code
base is built into a separate Linux/amd64 image for each task so that the input
socket name and image modality remain fixed.

Build either image from the workspace root:

```bash
bash doserad_proton_ct/submission/build.sh proton-ct
bash doserad_proton_ct/submission/build.sh proton-mri
```

Create a portable model package after checkpoint selection:

```bash
python3 doserad_proton_ct/scripts/make_portable_checkpoint.py \
  --input CHECKPOINT.pt --output doserad_proton_ct/dist/proton-ct/best.pt
bash doserad_proton_ct/submission/package_model.sh \
  doserad_proton_ct/dist/proton-ct/best.pt \
  doserad_proton_ct/dist/proton-ct/model.tar.gz
```

Export a container for upload:

```bash
bash doserad_proton_ct/submission/save_image.sh proton-ct \
  doserad_proton_ct/dist/proton-ct/doserad2026-proton-ct.tar.gz
```

Runtime environment overrides:

- `MODEL_CHECKPOINT`: relative checkpoint path under `/opt/ml/model`.
- `INFERENCE_PATCH_SIZE`: comma-separated `Z,Y,X`, default `128,128,128`.
- `INFERENCE_OVERLAP`: sliding-window overlap, default `0.25`.
- `INFERENCE_BATCH_SIZE`: conditions processed per GPU batch, default `8`.
- `BEAMLET_CHUNK_SIZE`: beamlets predicted per GPU chunk, default `32`. This
  bounds VRAM, not host memory: finished dose maps are streamed straight into
  their final byte offset in the output MetaImage, so host memory stays at one
  frame regardless of how many maps a run contains.
- `RAY_GATE_THRESHOLD`: physical-ray gate; defaults to at least `1e-6`.
- `TORCH_COMPILE=0`: disable compiled inference.
- `WARMUP_MODEL=0`: disable model warm-up.
- `SKIP_EMPTY_RAY=0`: disable empty-ray patch pruning.
- `MASK_OUTSIDE_BODY=0`: disable body masking.
- `DISABLE_AMP=1`: disable mixed precision.

Repeat with `proton-mri` and a separate model package for Task 4. Upload each
container under its matching algorithm's **Containers** page and its
`model.tar.gz` under **Models**.

The final checkpoint must be chosen using patient-separated full-volume
validation; a local smoke test does not replace preliminary A10G evaluation.

## Output memory contract

Dose maps are written with a streaming 4D MetaImage writer rather than by
buffering every map and calling `sitk.JoinSeries` at the end of the run. On the
largest training grid a single map is 170 MiB, so a 500-beamlet run would need
about 45 GiB just to hold the frames and roughly twice that at the moment
`JoinSeries` copies them, against at most 31 GiB of usable DRAM on the
`ml.g5.2xlarge` A10G instance. The writer keeps one frame in memory, seeks to
`header + idx_in_output * frame_bytes` and writes there, which also makes frame
placement independent of the order beamlets finish in.

The header is produced by SimpleITK itself from a 1x1x1x1 probe stack and only
`DimSize` is patched, so byte-for-byte output is unchanged. This is pinned by
`tests/test_submission_output_stream.py`, which also checks the writer against
the real geometry of the training set.
