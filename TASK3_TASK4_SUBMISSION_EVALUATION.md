# DoseRAD2026 Task 3/4 — evaluation and submission set

> **Snapshot of 2026-08-15, superseded.** The checkpoints assessed below belong to the
> end-to-end dose-prediction networks that were the submission candidates at that date.
> They were later replaced by the hybrid synthetic-CT + analytic pencil-beam method,
> which is what was submitted to the final test phase. See the README for the final
> method and results. Kept as a record of the intermediate stage.

Local freeze date: 2026-08-15 (Asia/Bangkok).

## Conclusions

- Task 3 Proton CT and Task 4 Proton MRI have finished training, checkpoint selection,
  overfit checks and full-volume evaluation, and are packaged as two separate algorithms.
- Both final images pass the offline smoke test: `/health = 200`, `/invoke = 201`, all
  10 scalar-4D MHA outputs present, and the correct `linux/amd64` architecture.
- The local results are enough for a **first preliminary submission**, but no claim about
  a top placement can be made before hidden-test scores exist. The official site still
  carries the IDD error warning and asks participants not to submit if they want to keep
  their attempt.

## Data and leakage

- The 75 public patients are split at patient level and held fixed: 60 train, 15 validation.
- No patient appears in both train and validation.
- Full-volume evaluation uses 45 beamlets spread across all 15 validation patients.
- Validation was used to select checkpoints and parameters, so only the preliminary
  hidden test is a fully out-of-sample assessment.

## Selected checkpoints

| Task | Primary checkpoint | MAE | NRMSE | IDD | Scale ratio |
|---|---|---:|---:|---:|---:|
| Task 3 — Proton CT | blend 50% v1 + 50% v2 | 0.048287 | 0.006003 | 0.197855 | 0.950037 |
| Task 4 — Proton MRI | v1 teacher-student | 0.047828 | 0.005971 | 0.194345 | 0.944737 |

These figures are full-volume validation at `patch=128³`, `overlap=0.25`,
`ray_gate=1e-6`, with no relative cutoff.

### Overfit control

- Task 3 v1 completed 30 epochs; train and validation still move together, with no clear
  sign of classical overfitting.
- Task 3 v2 improved IDD by 5.30% but worsened MAE by 3.42%; epoch 2 made validation
  worse again, so it was stopped early. The blend was chosen to balance the risk.
- Task 4 v1 completed 30 epochs and is the primary checkpoint.
- Task 4 v2 improved IDD by 4.12% but worsened MAE by 3.75% and scale by 0.79%; it was
  not used as the primary submission.

## Runtime

Proton conditioning has been vectorized on GPU and cross-checked against the NumPy
implementation:

- largest tensor difference: `1.19e-7`;
- mean full-volume difference: `1.50e-10`;
- `overlap=0.25` retained, accuracy configuration unchanged.

Benchmarked on a real validation image of `120×447×449` with 64 conditions spread evenly:

| Task | Estimated inference for 500 beamlets | Peak CUDA at 64 beamlets |
|---|---:|---:|
| Task 3 | 428.69 seconds | 22.99 GiB reserved |
| Task 4 | 427.50 seconds | 22.99 GiB reserved |

The submission uses `BEAMLET_CHUNK_SIZE=32`; that configuration measured a peak of about
17.14 GiB, leaving headroom on a 24 GiB A10G. The 428–429 second figures are inference
estimates on an NVIDIA GB10, not the official runtime, which Grand Challenge fits on an
A10G, and they do not account for every I/O and hardware difference. The official proton
threshold is 500 seconds.

## Final archives

### Task 3 — Proton CT

- File: `doserad_proton_ct/dist/proton-ct/doserad2026-proton-ct.tar.gz`
- Size: 3,934,305,172 bytes
- SHA-256: `b84bda60bb88a8b4f2017a6c6738c90ec42db68953511a599bb02cc036adea26`
- Model SHA-256: `b4663434dc763c280c7c2b4b4241a6bbfe48c5ba7de57a2e998373cf4c14b482`
- Docker image: `sha256:399d0ee12b4c9ee4351a5b6c141765369bc27d6b3f2dfdfef1f7e51028a0a008`

This archive replaces the `82b08e4c…` build of 2026-08-15, after the output-writing fix
(see "Host memory fix" below). The checkpoint is unchanged.

### Task 4 — Proton MRI

- File: `doserad_proton_ct/dist/proton-mri/doserad2026-proton-mri.tar.gz`
- Size: 3,934,315,956 bytes
- SHA-256: `9099c2eed0c2304e3ca8f975dc3523beae69cb1a94c5516be1c3f87a58f627b7`
- Model SHA-256: `ffddd2a90642bc9bc8549b21e7434ecedfe8bba14f90650a918b787188e2fa4c`
- Docker image: `sha256:a0a442c0409caa31decb7aa36b3e0e5500e96de4ecdf49e5a4e2827c4ff2bafd`

This archive replaces the `dcfe6bb6…` build of 2026-08-15. The checkpoint is unchanged.

Both archives pass `gzip -t`, are `linux/amd64`, and carry
`org.grand-challenge.api-method=invoke` with `TASK=proton-ct` and `TASK=proton-mri`
respectively.

## Host memory fix (2026-08-16)

The previous build held every dose map of a run in RAM and only called
`sitk.JoinSeries` at the end. Measured on the real validation grid `120×447×449`
(91.9 MiB per map), 64 maps:

| Writing strategy | Peak RSS | Output |
|---|---:|---:|
| Buffer + JoinSeries | 12.143 GiB | 6,165,596,492 bytes |
| Streaming writer | 0.660 GiB | 6,165,596,492 bytes |

The buffered version peaks at roughly 2.1× the size of the stack, because `JoinSeries`
creates an additional full copy. Extrapolated to 500 beamlets: 44.9 GiB for the frames
alone, so about 90 GiB at the moment of the copy. The Grand Challenge A10G instance
allows at most 31 GiB of usable DRAM (`ml.g5.2xlarge`, after 1 GiB reserved), and
`ml.g5.xlarge` only 15 GiB. The streaming version writes each frame straight to offset
`header + idx_in_output * frame_bytes`, so the peak is a single frame and does not depend
on the order in which beamlets finish.

The output is unchanged: the header is generated by SimpleITK itself from a 1×1×1×1
probe, with only `DimSize` patched. Verified byte-identical against `JoinSeries` on
**all 75** real CT geometries of the training set (0 mismatches), together with the 17
tests in `doserad_proton_ct/tests/`.

The container gate was re-run offline after the fix, for **both tasks**
(`submission/smoke_test.sh proton-ct 4` and `... proton-mri 4`): `/health = 200`,
`/invoke = 201`, slot 1 a 4D stack of exactly 4 frames, and the remaining 9 slots
1×1×1×1 placeholders.

## Further issues found during review

**The smoke-test fixture had been feeding garbage into the container.**
`create_submission_smoke_fixture.py` wrote MHA files with `CompressedData = True` but
without `CompressedDataSize`. MetaIO printed "Uncompress failed" and then returned
uninitialized memory instead of raising, so input images carried values up to 1.8e+36.
Every earlier smoke test therefore proved only the plumbing, never the arithmetic. The
fixture now writes uncompressed.

**The fixture's MRI phantom was degenerate.** The body mask for MRI is
`image <= bounds[0]`, where `bounds[0]` is the 0.5 percentile of the positive voxels.
The old phantom set the entire foreground to exactly 1.0, so `bounds[0] = 1.0` and the
mask erased the whole volume: the Task 4 smoke test returned `peak_dose = 0` regardless
of what the model did. On real validation MRI the mask removes only a further 0.14–0.34%
beyond the background (`1ABB039` 0.336%, `1ABB041` 0.136%, `1ABB042` 0.156%), so this was
a fault in the fixture, not in inference. With a gradient phantom instead, Task 4 gives
`peak_dose = 7.14e-05`, comparable to Task 3 (`7.61e-05`).

**The cutoff could zero out an entire dose map.** Raw predictions peak at 7.61e-05
(Task 3) and 7.14e-05 (Task 4), while the `minimum_cutoff` in the specification's example
is 0.02 — roughly 260–280 times larger. The checkpoint's `dose_scale` is 1.1307e-03 and
an audit of the training set puts the per-map maximum between 4.84e-04 and 1.64e-03, so
if the hidden test applies a cutoff of 0.02 in the same units, every map becomes 0.
The container still follows the specification (zeroing below the cutoff is mandatory),
but it now prints a warning with the pre-cutoff maximum and the `peak_dose` of the whole
run, so that a preliminary log exposes a unit mismatch immediately instead of silently
submitting all zeros. The training metadata contains no `minimum_cutoff`, so this cannot
be settled with local data.

## Assessment of competitiveness

Local validation shows two stable models, and Tasks 3 and 4 clear every local technical
gate. A top placement still cannot be inferred, because the leaderboard also uses
plan-level MAE, local gamma 1%/1 mm, the DVH score and A10G runtime, and none of these
have hidden ground truth that can be reproduced locally. The first preliminary attempt
should use the two primary archives above, once the IDD warning on the official site is
gone. The preliminary result will decide whether the next attempt goes to the IDD-leaning
checkpoint or to further accuracy/runtime optimization.
