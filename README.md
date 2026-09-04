# DoseRAD2026 — Task 3 and Task 4

Code for my submissions to **DoseRAD2026 — Real-time Dose Calculation in Radiotherapy**,
hosted on Grand Challenge as part of MICCAI 2026.

Author: **Ngoc Diep Thai** — Hanoi–Amsterdam High School for the Gifted, Hanoi, Vietnam
Grand Challenge profile: <https://grand-challenge.org/users/thaingocdiep/>
Solo participant, no lab or institutional affiliation.

## What is in this repository

| Directory | Task | Final-phase submission |
| --- | --- | --- |
| `doserad_proton_ct/` | Task 3 — proton dose on CT | Yes — *Proton CT Dose Engine v1* |
| `doserad_proton_mri/` | Task 4 — proton dose on MRI | Yes — *Proton MRI Dose Engine v1* |
| `doserad_photon_ct/` | Task 1 — photon dose on CT | No — development only; the container did not finish within the evaluation runtime limit |
| `doserad_photon_mri/` | Task 2 — photon dose on MRI | No — development only |

All four tasks live in one repository because they share most of the pipeline: the same
pyRadPlan engine wrapper, beamlet handling, worker pool and container packaging. Splitting
them would duplicate the shared code and force anyone reproducing the results to read
several places instead of one. `doserad_proton_mri/` reuses the implementation in
`doserad_proton_ct/` and adds the MRI-specific stage.

The `run_*.sh` scripts at the repository root are the job scripts used during the
competition to queue training and evaluation runs on a single workstation. They are kept
as they were rather than cleaned up, so that the exact sequence behind each reported
result stays visible.

## Results — final test phase, published 1 September 2026

| Task | Placement | Notes |
| --- | --- | --- |
| Task 4 — proton dose on MRI | 6th of 9 finalist teams | 4th on DVH-based clinical score, 4th on IDD curve distance |
| Task 3 — proton dose on CT | 11th of 13 finalist teams | evaluation runtime 261 s |

Both containers ran on the organizers' evaluation infrastructure with zero implementation
errors. Official leaderboards:
[Task 4](https://doserad2026.grand-challenge.org/evaluation/final-testing-proton-dose-on-mr/leaderboard/) ·
[Task 3](https://doserad2026.grand-challenge.org/evaluation/final-testing-proton-dose-on-ct/leaderboard/)

## Method

### Task 3 — proton dose on CT

Purely analytic, with no learned component. The open-source pyRadPlan analytic proton
pencil-beam engine computes each beamlet's dose directly on the planning CT, which supplies
mass density through the challenge-provided Hounsfield-to-density lookup table. Generic
proton machine model, air-offset correction, 25 mm geometric lateral cutoff.

Two changes carried most of the runtime improvement, from 621 s to 261 s on the evaluation
platform:

- each beamlet is computed on a transverse slab of ±12 slices around its isocenter, with
  the full transverse plane retained;
- beamlets that share an isocenter slice are grouped into a single dose-influence
  computation, on a 2 mm in-plane dose grid rather than the CT's native 1 mm.

Cropping the transverse plane to a corridor around the ray was tried and rejected: it
corrupts the engine's own body segmentation and air-offset correction, so the
water-equivalent depth becomes wrong and thoracic beamlets silently lose most of their
dose.

### Task 4 — proton dose on MRI

<!-- VERIFY: this paragraph must describe the code path that produced the submitted
     container, and must match the LNCS report. Rewrite if the folder contains a
     different approach. -->

A 3D residual U-Net (31.4 M parameters) trained on the challenge data, followed by the same
analytic pencil-beam engine used in Task 3. Diagnosing an incorrect density representation
on MRI — found by comparing per-metric positions on the leaderboard rather than in the code —
reduced IDD curve distance from 0.295 to 0.024 in the preliminary phase.

The full method for both tasks is described in the LNCS-format report submitted to the
organizers.

## Reproducing the results

Challenge data is **not** included in this repository and is not redistributed here; it is
available from the DoseRAD2026 organizers under the challenge's own terms.

Environment:

```
Python      TODO
PyTorch     TODO
pyRadPlan   TODO
```

```bash
# install
TODO

# Task 3 — inference on a prepared case
TODO

# Task 4 — training, then inference
TODO

# build a submission container
TODO
```

Containers were cross-built for x86 evaluation from an ARM64 workstation (NVIDIA DGX Spark).

## Challenge

DoseRAD2026 concluded on 2 September 2026 with 46 active participants and 565 algorithm
submissions, 59 of them in the final test phase. Organized by the German Cancer Research
Center (DKFZ), the Paul Scherrer Institute, Delft University of Technology, Amsterdam
University Medical Center and University Medical Center Utrecht, among others.
