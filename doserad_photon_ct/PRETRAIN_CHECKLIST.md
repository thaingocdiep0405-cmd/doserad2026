# Photon-CT pre-training checklist

The software pipeline below is implemented. Items that depend on the complete
75-patient download or a final trained checkpoint still have to be executed.

## 1. Data integrity

- Wait for all 75 public training patients.
- Run `bash scripts/finalize_dataset.sh`.
- Require every patient to have CT, metadata JSON and every dose file implied by
  `beam_idx` and `cp_idx`.
- Re-run `python3 scripts/create_splits.py` only after the final audit. Never
  split individual control points from the same patient across train and
  validation.

## 2. Data representation

Each training target is a single control-point dose volume. Its conditioning
information is:

- one patient CT volume;
- isocentre and source-axis distance from the beam;
- gantry angle from the control point;
- left/right MLC leaf positions from the control point.

The CT is reused by hundreds of targets. Keep one CT cache entry per patient
instead of writing a duplicate CT beside every target.

The implemented neural baseline converts beam metadata to CT, body, MLC
aperture, depth, lateral and superior-inferior channels. Aperture coordinates
are projected through SAD to model beam divergence and use the same MLC
orientation transform as the official pyRadPlan baseline.

## 3. Spatial preprocessing

- Photon CT and inspected dose maps share 2 x 2 x 2 mm spacing and the same
  origin, direction and size. Preserve that geometry in predictions.
- Patient shapes vary, so batching needs deterministic crop/pad or patch-based
  sampling. Store the inverse crop/pad transform and reconstruct the original
  grid before writing a prediction.
- Do not resample data that is already on the common 2 mm spacing unless a
  measured model/memory trade-off justifies it.
- Bias training patches toward non-air CT and non-zero dose regions; include
  some background patches so false positive dose is still penalised.

These crop/pad, positive-patch and reconstruction paths are implemented and
covered by tests.

## 4. Intensity and target normalisation

- CT background is `-1024` HU in the inspected samples. Determine clipping
  limits from the complete train split, then apply one fixed transform to train,
  validation and inference.
- Individual control-point doses are sparse and approximately `1e-5` to `1e-4`
  in the current sample. Recompute full training statistics before fixing a
  scale.
- Use one fixed dose scale learned from the training split and invert it at
  inference. Per-volume max normalisation would discard absolute output scale.
- Start with a dose-aware loss that gives non-zero/high-dose voxels sufficient
  weight, while retaining an error term over the full grid.

The training CLI currently defaults to fixed `1e-4` target scaling, CT clipping
to `[-1024, 2000]`, Smooth-L1 over the full patch, high-dose L1 and a spatial
gradient term. Reconfirm these values after final statistics.

## 5. Baselines and metrics

Run two baselines before model training:

1. A zero-dose output to verify the local evaluator and file contract.
2. The official pyRadPlan SVDPB photon baseline for accuracy/runtime reference.

The official evaluator includes:

- masked beam MAE;
- integrated depth-dose curve distance;
- stratified plan MAE;
- local 3D gamma at 1% / 1 mm;
- PTV/OAR DVH scores;
- runtime measured separately.

The public patient folders currently expose CT/MR, beam JSON and individual
dose maps, but no public `weight.json` or structure set was observed. Therefore
beam-level validation can be implemented immediately; exact plan/DVH validation
requires the corresponding plan weights and structures. Do not fabricate them.

`scripts/evaluate_checkpoint.py` performs full-volume masked beam MAE/NRMSE.
Use the cloned official evaluator for final challenge scoring when the required
plan metadata and structures are available.

## 6. Submission compatibility

Training JSON and submission JSON are related but not identical. The submission
adapter must handle stacked metadata and its `output_info` mapping, then place
each predicted 3D dose in the requested output stack/index.

The official container contract requires:

- `/health` returns HTTP 200 after the model has loaded;
- `/invoke` writes all ten output slots and returns HTTP 201;
- unused output slots contain a valid placeholder `.mha`;
- model weights load once at startup;
- Docker label `org.grand-challenge.api-method="invoke"`;
- no internet dependency at inference time.

The development machine is `aarch64`, while the official template uses an
`linux/amd64` CUDA base image. Keep training code architecture-neutral and plan
an amd64 build/test path for the final Grand Challenge container.

The adapter and Dockerfile are implemented under `submission/`. Before upload,
run an official fixture end-to-end, benchmark every control point in a job, and
tune sliding-window batch size/overlap to stay inside the platform timeout.
