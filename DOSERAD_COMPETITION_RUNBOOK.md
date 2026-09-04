# DoseRAD2026 competition runbook

## Current training queue

Task 1 and Task 2 run sequentially through:

```bash
./run_doserad_tasks_1_2.sh
```

The active order is deliberate: Task 1 uses the GPU first; Task 2 starts only
after Task 1 exits successfully. Follow progress with:

```bash
tail -f doserad_photon_ct/runs/photon_ct_v3_physics/train.log
tail -f doserad_photon_ct/runs/photon_ct_v3_physics/metrics.jsonl
tail -f doserad_photon_mri/runs/photon_mri_v3_teacher_student/train.log
tail -f doserad_photon_mri/runs/photon_mri_v3_teacher_student/metrics.jsonl
```

## Task 1 candidate

Photon CT v3 warm-starts from the v2 MAE/IDD blend. It adds mass density,
signed aperture edge, inverse-square attenuation, local field width and a
primary-fluence prior. New input weights start at zero, so the first v3 forward
pass reproduces the learned v2 behavior before adapting.

The objective combines whole-body Smooth-L1, high-dose error, spatial-gradient
error, the official normalized masked beam MAE, a differentiable directional
IDD surrogate and dose-scale calibration. Full validation samples 15 patients
and three control points per patient across beams and arc locations.

## Task 2 candidate

Photon MRI v3 warm-starts from the completed MRI baseline. It adds the same
beam-physics priors and an auxiliary synthetic-density head. Paired public CT
is converted using the official HU-to-mass-density calibration and supervises
that head during training only. The submitted model still receives MRI and
beam metadata only; CT is not required at inference.

## Checkpoint promotion gate

Do not promote the final epoch automatically. For each task, evaluate
`best_full.pt`, `best_idd.pt` and `last.pt` on the identical fixed cohort:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint runs/EXPERIMENT/CHECKPOINT.pt \
  --max-records 75 --batch-size 8 --overlap 0.25 \
  --skip-empty-aperture --mask-outside-body --torch-compile --pad-batch \
  --device cuda --output artifacts/CANDIDATE.json
```

Promotion requires:

- lower patient-aggregated masked beam MAE and IDD with bootstrap intervals;
- no material regression in NRMSE;
- stable performance across patients, beams and gantry-angle quantiles;
- all unit tests and the offline amd64 container fixture passing;
- official A10G runtime below the safety budget, including model loading and I/O.

If best-MAE and best-IDD are complementary, test fixed weight blends at 25/75,
50/50 and 75/25 and keep one model. An ensemble is allowed only if measured
runtime stays safe.

## Limits of local validation

The public training folders do not contain the hidden plan weights and
structures used for exact plan-MAE, gamma and DVH scoring. Local beam MAE/IDD
therefore reject weak models but cannot establish a leaderboard rank. The
preliminary evaluation is the authoritative six-metric and runtime gate.

Do not spend a submission slot while the organizer has an active evaluator
warning. When submissions are reliable, use slots for controlled one-change
ablations and archive the image digest, checkpoint hash, configuration and all
reported metrics for every attempt.
