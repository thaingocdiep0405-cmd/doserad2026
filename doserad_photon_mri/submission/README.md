# Photon-MRI submission adapter

This adapter implements the official Grand Challenge `/health` and `/invoke`
contract for the trained `PhotonDoseUNet3D` checkpoint. It reads stacked photon
metadata, maps every control point to its requested output stack/index, predicts
on the original MRI grid, applies `minimum_cutoff`, and writes all ten output
slots.

Build the amd64 image from the project root:

```bash
bash submission/build.sh
```

Save the built image for upload:

```bash
bash submission/save_image.sh dist/doserad2026-photon-mri.tar.gz
```

Package the best checkpoint as a Grand Challenge model resource:

```bash
bash submission/package_model.sh \
  dist/best.pt dist/model.tar.gz
```

Runtime environment overrides:

- `MODEL_CHECKPOINT`: relative checkpoint path under `/opt/ml/model`.
- `INFERENCE_PATCH_SIZE`: comma-separated `Z,Y,X` dimensions.
- `INFERENCE_OVERLAP`: sliding-window overlap, default `0.25`.
- `INFERENCE_BATCH_SIZE`: number of patches per GPU batch.
- `CONTROL_POINT_CHUNK_SIZE`: control points retained per host-memory chunk,
  default `64` after local peak-memory validation.
- `TORCH_COMPILE=0`: disable the default compiled inference graph.
- `TORCH_COMPILE_FALLBACK=0`: fail instead of automatically reverting to eager
  inference when compiled warm-up is not supported by the grading GPU.
- `PAD_INFERENCE_BATCH=0`: disable padding partial batches to the compiled size.
- `WARMUP_MODEL=0`: disable CUDA/compiled-graph warm-up during model loading.
- `SKIP_EMPTY_APERTURE=0`: disable empty-aperture patch pruning.
- `MASK_OUTSIDE_BODY=0`: disable forcing MRI zero-background dose to zero.
- `DISABLE_AMP=1`: disable CUDA mixed precision.
- `VERBOSE_CONTROL_POINTS=1`: print one log line per control point; disabled by
  default to avoid distorting runtime.

The selected checkpoint is `runs/photon_mri_baseline/best_idd.pt`. The
full/IDD blends improved MAE but degraded IDD and did not dominate this model.
The adapter uses batch 8, chunk 64, compiled inference, model warm-up,
multi-control-point batching and GPU-side sliding-window blending.
