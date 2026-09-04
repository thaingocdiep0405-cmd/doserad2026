# Photon CT leaderboard optimization roadmap

Updated: 2026-08-13

## Objective and constraints

The official score is a weighted mean of ranks, not a single accuracy value.
We therefore optimize beam MAE, IDD, plan MAE, local 1%/1 mm gamma, DVH, and
runtime together. Runtime has double rank weight, and photon submissions at or
above 181 seconds are excluded.

Do not submit while the official site displays the current IDD-bug warning.
There are only 10 preliminary and 2 final submissions per task.

## Reproducible validation protocol

- Patient-level split: 60 train / 15 validation; no control point leakage.
- Patch validation crops are deterministic across epochs.
- Full validation: one fixed control point from every validation patient.
- Save independent best checkpoints for patch MAE, full-volume beam MAE, and
  IDD. Never compare patch MAE and full-volume MAE on the same checkpoint scale.
- Before submission, evaluate candidate checkpoints on a larger fixed cohort
  sampled uniformly by patient, anatomy, beam and gantry range.

## Baseline measured on the fixed 15-patient cohort

`runs/photon_ct_baseline/last.pt`, patch 128, overlap 0.25:

- beam MAE: 0.0832268
- IDD distance: 0.3011602
- NRMSE: 0.0186833
- 2.93 seconds per control point when evaluated sequentially

With empty-aperture pruning and dose forced to zero outside the CT body:

- beam MAE: 0.0832337 (effectively unchanged)
- IDD distance: 0.2420431 (19.6% lower)
- NRMSE: 0.0182682
- 2.63 seconds per control point

Multi-control-point inference, batch 8 and GPU-side patch blending, measured on
16 maps from one CT, estimates 157.4 seconds for 181 maps on the local GB10.
This is below the official limit but must still pass the official A10G container
fixture including I/O.

## V2 changes now implemented

- Warm-start from the completed 100-epoch baseline.
- Add official HU-to-mass-density calibration as a seventh input channel.
- Define the loss high-dose mask using 10% of the *whole beam* maximum, matching
  the evaluator, instead of 10% of each cropped patch maximum.
- Exclude CT padding/air outside the body from the full-dose loss.
- Evaluate fixed full volumes every two epochs and track MAE plus IDD.
- Keep `best_full.pt` and `best_idd.pt` separately.

Run with:

```bash
bash scripts/train_v2_gpu.sh
```

## Experiment ladder

Only advance a candidate if it improves the fixed validation cohort.

1. V2 density fine-tune, 30 epochs (currently configured).
2. Compare `best_full.pt` and `best_idd.pt` with the same inference parameters.
3. If v2 improves the geometric mean of normalized MAE and IDD, test
   high-dose weights 2 and 6 for 10 epochs each, warm-started from baseline.
4. Test positive-patch probabilities 0.7 and 0.9 for 10 epochs each.
5. Promote only the best setting to a longer 50-epoch fine-tune.
6. If two seeds provide complementary errors, distil them into one student;
   do not deploy a two-model ensemble unless it stays comfortably below the
   runtime limit.

## Submission gate

- All unit tests pass.
- Container runs without network and writes all ten output slots.
- Output uses genuine 4D `JoinSeries`, correct ordering and identical geometry.
- Apply cutoff with `prediction <= minimum_cutoff`, not only `<`.
- Official/local fixture runtime is below 160 seconds to retain safety margin.
- No NaN/Inf and no nonzero value at/below the declared cutoff.
- Submit first to preliminary after the IDD warning disappears.
- Record all six metrics and runtime for every preliminary attempt; use the
  remaining slots for one-variable ablations, not repeated identical models.

No local metric can guarantee a leaderboard position. Hidden-test Mean Position
exists only after a valid preliminary submission.

## Final v2 candidate (2026-08-13)

Training completed for all 30 fine-tune epochs. A single-model weight blend of
`best_full.pt` and `best_idd.pt` at 50/50 is selected because it improves both
local primary beam metrics without paying the runtime cost of an ensemble.

Fixed 15-patient evaluation, patch 128 and overlap 0.25:

| Candidate | Beam MAE | IDD | NRMSE |
| --- | ---: | ---: | ---: |
| Baseline + pruning | 0.083234 | 0.242043 | 0.018268 |
| v2 best-full | **0.077123** | 0.249566 | 0.018553 |
| v2 best-IDD | 0.083667 | **0.223450** | 0.018516 |
| v2 blend 25/75 | 0.081508 | 0.225872 | 0.018324 |
| **v2 blend 50/50** | **0.079682** | **0.231776** | **0.018301** |
| v2 blend 75/25 | 0.078194 | 0.240016 | 0.018389 |

Relative to the pruned baseline, the 50/50 candidate improves Beam MAE by
about 4.3%, IDD by about 4.2%, and keeps NRMSE effectively unchanged. Local
batched inference measured 114.8 seconds for the standardized 181-map estimate
on GB10. The official A10G runtime remains the authoritative value.

## Compiled inference update (2026-08-14)

The selected checkpoint remains the v2 50/50 blend. Inference now uses batch 8,
chunk 64, `torch.compile`, matching FP32-input warm-up under autocast, and pads
the final partial batch to avoid compiling a second graph inside `/invoke`.

- 64-control-point benchmark: 29.61 seconds on local GB10;
- projected 181-control-point runtime: 83.73 seconds;
- peak CUDA memory: 5.64 GiB allocated, 11.22 GiB reserved;
- fixed 15-patient MAE: 0.079718;
- fixed 15-patient NRMSE: 0.018312;
- fixed 15-patient IDD: 0.232190.

Relative to eager validation, compiled inference changes MAE by 0.045%, NRMSE
by 0.058%, and IDD by 0.179%, while the standardized runtime projection drops
from 114.8 to 83.7 seconds. The official A10G measurement remains required.

Container smoke validation passed:

- linux/amd64 image with required `invoke` label;
- Python 3.11 portable checkpoint loads successfully;
- `/health` returns 200 and `/invoke` returns 201;
- all ten output slots are genuine scalar 4D MHA files;
- no NaN/Inf and cutoff validation passes.

Prepared artifacts:

- inference checkpoint: `dist/best.pt`;
- model resource: `dist/model.tar.gz`;
- container image: `dist/doserad2026-photon-ct.tar.gz` (after `save_image.sh`).

## V3 physics-prior fine-tune (2026-08-14)

V3 is implemented and running from the v2 50/50 blend. It adds five explicit
physics channels (for 11 total), competition-aligned MAE/IDD/scale loss terms,
patient-balanced full validation, and numerically stable exclusion of
near-empty patches from the IDD surrogate. A load test with batch 4 and
96-cubed patches completed on the local GB10 without OOM or skipped optimizer
steps. See `DOSERAD_COMPETITION_RUNBOOK.md` (repo root) for monitoring
and checkpoint promotion.

## V6 radiological-depth + capacity (2026-08-17)

Gap analysis against the preliminary leaderboard located the 8x Beam-MAE
deficit in (1) missing ray-traced attenuation information, (2) low-dose
scatter placed wrongly (IDD 30-60x worse than the leaders), (3) model
capacity, and (4) ~9% recoverable per-CP scale error.

V6 implements the first and third items:

- `src/doserad_photon_ct/radiological.py`: torch-based water-equivalent
  depth. Per z-slice: rotate the (2x-downsampled) density slice so the
  beam runs along one axis, midpoint-rule cumsum, rotate back. Odd canvas
  with integer margins keeps 90-degree-multiple gantry angles exactly on
  the pixel grid (no bilinear blur accumulating through the cumsum).
  Axis-aligned cases validate to <0.1 mm at full resolution.
- Two new conditioning channels (13 total): normalized radiological depth
  and analytically attenuated primary fluence
  (`exp(-0.0046/mm * max(depth - 15 mm, 0))` on top of the v3 prior).
- Dataset computes the depth only for the patch z-slab (+~90 ms/item,
  ~18% over v5); inference computes one full volume per control point on
  GPU. Train/val/full-val/evaluate/predict/submission adapter all thread
  the new `radiological_depth` model-config flag.
- `scripts/train_v6_radiological.sh`: base24-L5 (31.4M params), otherwise
  the v5 recipe. Runtime on A10G is the open risk: benchmark the 181-CP
  projection before packaging; fall back to base16-L5 with the same
  channels if it exceeds the safety margin.
