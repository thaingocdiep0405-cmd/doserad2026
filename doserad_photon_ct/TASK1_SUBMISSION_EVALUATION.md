# Task 1 submission evaluation

Updated: 2026-08-15

## Decision

Task 1 has a strong Level-1 candidate, but it is not yet cleared for final
submission. Use `runs/photon_ct_v3_physics/best_idd.pt` as the primary
preliminary-test candidate and `runs/photon_ct_v3_physics/last.pt` as the
secondary candidate. Do not use any of the tested weight blends as the primary
candidate: none improves the measured MAE--IDD balance over `best_idd.pt`.

Final submission clearance still requires:

1. a clean A10G-compatible runtime benchmark below the 181-second photon hard
   limit, with adequate margin;
2. preliminary-leaderboard results for both primary and secondary candidates;
3. validation of plan-level MAE, 1%/1 mm gamma and DVH on data containing plan
   weights and structures. Those labels are not present in the public local
   validation subset.

The end-to-end Docker gate is complete. The exported Linux/amd64 image loads
the portable `best_idd.pt` checkpoint without network access, returns HTTP 200
from `/health` and HTTP 201 from `/invoke`, and writes all ten required 4D MHA
output slots. This smoke test was run through amd64 emulation on the local ARM
host; it validates the interface and packaging, not full-case A10G runtime.

The primary checkpoint was benchmarked again on 2026-08-15 with 64 control
points, batch 8, chunk 64, 128-cubed patches, 0.25 overlap and compiled AMP
inference. The local GB10 projection is 127.40 seconds for 181 control points,
with 5.89 GiB allocated and 12.35 GiB reserved. This is 29.6% below the photon
hard limit, but the official fitted A10G runtime remains authoritative.

## Prepared preliminary-submission packages

- Container: `dist/doserad2026-photon-ct.tar.gz` (3,934,224,228 bytes),
  SHA-256 `d655854437adc9595d574fd6c7e1c72e0ae5a6afff5eab11174c452b5fa18a99`.
- Model: `dist/model.tar.gz` (7,256,530 bytes), SHA-256
  `e1addfe082c0f910dca5faf56e1e877a006e44947c86700ce92a82e9d85f3fe4`.
- Docker image: Linux/amd64, image ID
  `sha256:971a6f3914cbf94b42c9a1ff62d51486ff1ebc03c2685735cc44923458d18ae3`,
  with `org.grand-challenge.api-method=invoke`.

## Evaluation protocol

- Split: fixed patient-level validation split; no patient overlap with train.
- Coverage: 75 control points from 15 patients, five records per patient.
- Sampling: patient-balanced, beam-balanced and distributed across VMAT arc
  control-point quantiles.
- Inference: 128x128x128 sliding-window patches, 0.25 overlap, AMP, batch 4,
  empty-aperture pruning, body masking and padded final batch.
- Aggregation: records are averaged within each patient before patients are
  averaged, matching the official Level-1 aggregation rule.
- Uncertainty: paired patient bootstrap with 2,000 resamples and 95% confidence
  intervals.
- Metric parity: local beam MAE and IDD implementations are directly tested
  against the official evaluator.

This is a stratified validation sample, not a hidden-test or leaderboard score.

## Main results

Lower is better for all columns below. The relative score is a local diagnostic
geometric mean of MAE and IDD relative to v2; it is not the official
RankThenMean leaderboard score.

| Candidate | Masked beam MAE | IDD distance | NRMSE | Relative MAE--IDD |
|---|---:|---:|---:|---:|
| `v3_best_idd` | 0.081281 | **0.192842** | **0.017407** | **0.7628** |
| `v3_last` | **0.080275** | 0.196662 | 0.017440 | 0.7655 |
| `v2_10% + best_idd_90%` | 0.080952 | 0.198680 | 0.017487 | 0.7727 |
| `v3_full_25% + idd_75%` | 0.081068 | 0.198956 | 0.017525 | 0.7738 |
| `v2_10% + last_90%` | 0.080257 | 0.202021 | 0.017547 | 0.7758 |
| `v2_reference` | **0.078961** | 0.341157 | 0.018717 | 1.0000 |

Compared with the v2 reference, `v3_best_idd`:

- reduces IDD distance by 43.47%; the paired 95% CI for the absolute difference
  is [-0.21137, -0.09167], and all 15 patients improve;
- reduces NRMSE by 7.00%; paired 95% CI [-0.001616, -0.000988], and all 15
  patients improve;
- increases masked beam MAE by 2.94%; paired 95% CI [0.000382, 0.004138], so the
  MAE trade-off is statistically visible on this validation set.

Compared with the v2 reference, `v3_last`:

- reduces IDD distance by 42.35%, with all 15 patients improving;
- reduces NRMSE by 6.82%, with all 15 patients improving;
- increases MAE by 1.66%, but the paired 95% CI [-0.000199, 0.002861] includes
  zero. It is therefore the safer MAE-oriented alternative.

## Submission interpretation

The official ranking uses per-metric ranks, not the local relative score. It
includes masked beam MAE, IDD, plan MAE, gamma, DVH and runtime, with runtime
receiving double weight. Consequently, local Level-1 results cannot establish a
leaderboard position.

Recommended preliminary-test order:

1. `best_idd.pt`: best measured IDD and NRMSE, and best local balance.
2. `last.pt`: slightly weaker IDD but lower and statistically safer MAE.

Use the leaderboard breakdown to choose the final checkpoint. If the primary
candidate loses substantially on beam MAE or plan metrics, promote `last.pt`.
Do not select a candidate solely from the public Mean Position because that
position depends on the other submitted algorithms and can change.

## Reproducibility

Evaluation outputs are stored in:

- `artifacts/task1_candidate_evaluations/`
- `artifacts/task1_candidate_summary.json`

Commands:

```bash
bash scripts/evaluate_task1_candidates.sh
bash scripts/evaluate_task1_cross_version_blends.sh
python scripts/summarize_candidate_evaluations.py \
  --input-dir artifacts/task1_candidate_evaluations \
  --reference v2_reference \
  --output artifacts/task1_candidate_summary.json
```

