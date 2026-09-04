# DoseRAD2026 — Submission readiness audit

Audit date: 2026-08-14

> **Snapshot, superseded.** This is a self-assessment written at a point when no task was
> yet competitive. Its conclusions were acted on: the proton tasks were rebuilt around the
> hybrid synthetic-CT + analytic pencil-beam method described in the README, and Tasks 3
> and 4 reached the final test phase. Kept as a record of how the work was judged at the
> time, not as a description of the final submissions.

## Executive summary

At present no task has enough evidence to be called a contender. Tasks 1 and 2 have a
complete pipeline, checkpoints, containers and smoke tests, so they can be used for **one
diagnostic preliminary run** once the IDD error warning on the official site is lifted.
Tasks 3 and 4 are not ready: the proton data has not finished downloading and no
corresponding project, model or container exists yet.

Mean Position cannot be inferred from local validation. Mean Position only exists after an
algorithm has been scored on the hidden test set, across all six ranking dimensions.

## Official criteria used for this audit

Each task is ranked along six dimensions:

1. masked per-beam MAE;
2. IDD curve distance;
3. plan-level stratified MAE;
4. local gamma 1%/1 mm;
5. DVH clinical score;
6. standardized runtime.

Runtime is measured on an NVIDIA A10G 24 GiB and carries double weight in the ranking.
Photon entries are excluded if the standardized runtime reaches or exceeds 181 seconds;
proton entries are excluded at 500 seconds. Level-1 metrics must be aggregated
beam-to-patient and then patient-to-submission.

Official sources:

- [Metrics and ranking](https://doserad2026.grand-challenge.org/metrics-and-ranking/)
- [Tasks and getting started](https://doserad2026.grand-challenge.org/tasks-and-getting-started/)
- [Submission instructions](https://doserad2026.grand-challenge.org/submission-instructions/)
- [Timeline and rules](https://doserad2026.grand-challenge.org/timeline-and-rules/)
- [Final submission requirements](https://doserad2026.grand-challenge.org/final-submission-requirements/)

## Readiness per task

| Task | Data | Model / local validation | Container | Status |
| --- | --- | --- | --- | --- |
| 1 — Photon CT | 75/75, 40,500/40,500 maps | Yes; MAE 0.0797, IDD 0.2322 over 15 patients / 15 CP | Yes, smoke test passes | Diagnostic preliminary run only |
| 2 — Photon MRI | 75/75, 40,500/40,500 maps | Yes; patient-mean MAE 0.0862, IDD 0.3123 over 15 patients / 75 CP | Yes, smoke test passes | Diagnostic preliminary run only |
| 3 — Proton CT | Downloading; 44/75 complete cases, 48,352/81,000 maps at audit time | None | None | Cannot submit |
| 4 — Proton MRI | Shares the proton store, incomplete | None | None | Cannot submit |

### Task 1 — Photon CT

Strengths:

- patient-level split of 60 train / 15 validation, no control-point leakage;
- CT and density are both fed to the model;
- separate checkpoints for MAE and IDD, and candidate blends have been compared;
- inference is batched, chunked, AMP-enabled and compiled;
- container is correctly `linux/amd64`, uses the right invoke label, emits scalar 4D
  output, and passes the cutoff and NaN/Inf checks;
- 10/10 unit tests pass.

Falling short of a top-contender standard:

- only one control point per validation patient is evaluated; the 95% confidence interval
  on MAE is roughly 0.0690–0.0913, which is still wide;
- validation was used to select the checkpoint and the blend, so there is selection bias;
- at the end of training, patch training MAE is around 0.0668 against validation around
  0.0904 — a generalization gap of roughly 35%;
- the three plan-level metrics (plan MAE, gamma, DVH) have not been evaluated;
- the 83.7-second runtime estimate is from a local GB10, not the official A10G;
- the `torch.compile` path in the amd64 image cannot be GPU-tested on an ARM64 host.

The hidden leaderboard leaders are in the region of 0.0086–0.014 beam MAE. Local and
hidden test scores cannot be compared directly, but a gap of several times over is a
strong signal that the current model is only a technical baseline.

### Task 2 — Photon MRI

Strengths:

- clean patient split and complete data;
- evaluation over 75 records covering all 15 patients;
- model warm-starts from photon CT; inference is compiled, batched and chunked;
- local runtime estimate of 99.7 seconds for 181 maps, under the local hard limit;
- 8/8 unit tests pass.

Falling short:

- the patient-mean MAE, which is the metric the rules define, is 0.0862 — not the
  record-mean 0.08535;
- the 95% confidence interval on MAE is roughly 0.0769–0.0962, and IDD still varies widely;
- train/validation gap of about 34%, with the best checkpoint appearing early and
  degrading afterwards;
- MRI does not carry electron density directly, and the current model has no
  synthetic-CT / density-teacher branch or physics prior strong enough to compensate;
- the three plan-level metrics have not been evaluated, and there is no real A10G benchmark.

The hidden leaderboard leaders are in the region of 0.0158–0.0172 beam MAE. This gap, too,
is far too large to treat the current candidate as a competitive model.

### Tasks 3 and 4 — Proton

At audit time the download is still running. The proton store holds 45/75 patient folders,
44 complete cases and 48,352/81,000 dose maps. There is no `doserad_proton_ct` or
`doserad_proton_mri` yet, and no split, model, validation, benchmark or container.
Accuracy therefore cannot be assessed and neither task can be submitted.

Protons need a different physical model from photons: energy, Bragg peak,
water-equivalent path length, range uncertainty and lateral spot spread all have to be
conditioned on directly. Copying the photon U-Net and swapping the input is not a
sufficiently sound approach.

## Significant scientific gaps

1. **Validation does not represent the competition metrics.** Locally only two of the five
   accuracy metrics are covered; NRMSE is not a ranking metric.
2. **No independent holdout.** The same 15 patients are used to select the epoch, the
   blend and the inference parameters.
3. **The model leans on an image-to-image baseline.** The problem is physical transport
   with beam geometry, not merely 3D segmentation or regression.
4. **The loss does not track the ranking closely enough.** It needs masked normalized L1,
   a differentiable IDD term, a gradient/gamma surrogate, scale calibration, and a plan
   proxy where weights are available.
5. **Weak generalization.** There is no geometric augmentation consistent with the beam,
   no HU perturbation, and no systematic MRI bias/noise augmentation.
6. **Runtime is unconfirmed on A10G.** A CPU-emulated smoke test confirms the contract
   only, not CUDA, compilation or performance.

## Architecture proposed to become competitive

### Photon CT

Use a physics-prior residual network:

- ray tracing from the source through CT density to produce radiological depth;
- explicit aperture/MLC fluence, source distance and beam-axis coordinates;
- a primary-dose prior with attenuation and inverse-square falloff;
- a 3D network predicting the residual/scatter correction, with the output always
  constrained to be non-negative;
- a multi-objective loss tracking masked MAE + IDD + a dose-gradient/gamma surrogate.

### Photon MRI

Exploit the paired CT during training only:

- train a strong CT teacher first;
- give the MRI student an auxiliary synthetic-density/CT head;
- distil both features and dose from the CT teacher into the MRI student;
- add MRI bias field, intensity scaling, noise and artifact augmentation;
- final inference still receives MRI alone, as the task requires.

### Proton CT/MRI

- dedicated conditioning for energy and spot geometry;
- computed WEPL / radiological path length;
- priors for range/Bragg peak and lateral Gaussian spread;
- a beam's-eye-view representation as a sequence of slices; this suits forward transport
  and has been used in [DoTA](https://arxiv.org/abs/2202.02653), rather than forcing a
  global U-Net to relearn the entire geometry by itself;
- the network learns only the heterogeneity correction and the residual;
- for MRI, use a CT/density teacher during training, as in Task 2.

The MRI teacher / synthetic-density proposal is not merely a heuristic: MRI does not
directly supply the electron density that dose calculation requires, consistent with the
official task description and with the field's review of
[synthetic CT in MRI-only radiotherapy](https://pubmed.ncbi.nlm.nih.gov/34474325/).
The challenge dataset ships paired, registered CT–MRI, so cross-modality supervision is a
significant asset to exploit.

## Order of work with the highest probability of success

1. Wait for the proton download to complete; do not interrupt the running process.
2. Extend Task 1 validation into a fixed set with several control points per patient,
   stratified by anatomy, beam and gantry angle; report patient means and bootstrap CIs.
3. Build the physics-prior Photon CT model and run ablations one variable at a time.
4. Distil the CT teacher into Photon MRI.
5. Only once the proton data is complete at 75/75, build one shared proton physics core
   and then split it into CT and MRI heads.
6. Test the `linux/amd64` image on a real A10G; if compilation fails it must fall back
   safely to eager mode and still stay under the hard limit.
7. When the official IDD warning is gone, spend one preliminary slot on a stable baseline
   to obtain all six hidden metrics; do not spend slots on random attempts.
8. Use later slots for hypothesis-driven ablations, and lock the two final candidates
   before the deadline.

## Go/no-go gate before final submission

A task counts as final-ready only when all of the following hold:

- data and split audits pass, with no leakage;
- patient-level evaluation on a fixed cohort large enough to carry confidence intervals;
- no run of consecutive validation regressions while training continues;
- all five accuracy metrics covered, or corresponding preliminary feedback in hand;
- A10G runtime with at least a 20% safety margin against the hard limit;
- the container runs end-to-end over multiple images, with every output correct in
  geometry and cutoff;
- checkpoint, config, seed, commit/image digest and report all locked.

By this gate: Tasks 1 and 2 are not final-ready; Tasks 3 and 4 are not even
preliminary-ready.
