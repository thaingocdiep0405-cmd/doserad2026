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

### Repository layout

```
doserad2026/
├── doserad_proton_ct/                  # Task 3 (Proton CT) and shared proton code
│   ├── src/doserad_proton/             # Core library (~1800 lines)
│   │   ├── pencilbeam.py              #   pyRadPlan pencil-beam engine wrapper
│   │   ├── inference.py               #   Sliding-window + slab-based inference
│   │   ├── conditioning.py            #   10-channel conditioning (dose-net path)
│   │   └── data.py                    #   Manifest, splits, 81k pencil-beam dose maps
│   ├── scripts/                       #   Training, evaluation, benchmarking
│   │   ├── train_synthetic_ct.py      #   MRI→sCT network — trains the Task 4 model
│   │   └── train.py                   #   Dose-prediction network (superseded)
│   ├── submission/                    #   Dockerfile, FastAPI app, smoke tests
│   │   ├── inference_pb.py            #   Pencil-beam backend — Task 3 and Task 4
│   │   └── inference.py               #   Dose-network backend (superseded)
│   ├── paper/                         #   Task 4 LNCS report
│   ├── paper_t3/                      #   Task 3 LNCS report
│   ├── tests/                         #   Unit tests
│   └── artifacts/                     #   Evaluation metrics and benchmarks
│
├── doserad_proton_mri/                 # Task 4 (Proton MRI) — uses proton_ct code
│   └── artifacts/                     #   Task 4-specific evaluation results
│
├── doserad_photon_ct/                  # Task 1 (Photon CT) — development only
│   ├── src/doserad_photon_ct/          # Core library (~2250 lines)
│   │   ├── model.py                   #   3D residual U-Net architecture
│   │   ├── dataset.py                 #   Positive-dose-biased patch sampler
│   │   ├── conditioning.py            #   CT/body/MLC/beam conditioning (6 ch)
│   │   ├── inference.py               #   Sliding-window with Gaussian blending
│   │   ├── losses.py                  #   Dose-aware composite loss
│   │   ├── radiological.py            #   Radiological depth features
│   │   ├── metrics.py                 #   Beam MAE, IDD evaluation
│   │   ├── mha.py                     #   Compressed MHA reader/writer
│   │   └── dataset_index.py           #   One-row-per-control-point manifest
│   ├── scripts/
│   ├── submission/
│   ├── paper/                         #   Task 1 LNCS report
│   ├── tests/
│   └── artifacts/
│
├── doserad_photon_mri/                 # Task 2 (Photon MRI) — development only
│   ├── src/doserad_photon_mri/         # MRI-adapted library (~1980 lines)
│   ├── scripts/
│   ├── submission/
│   ├── tests/
│   └── artifacts/
│
├── LICENSE                                    # MIT
├── run_*.sh                                   # Job queue scripts (kept as-is)
├── DOSERAD2026_SUBMISSION_READINESS_AUDIT.md  # Self-assessment, 2026-08-14
├── DOSERAD2026_MASTER_ROADMAP.md              # Planning snapshot, superseded
├── DOSERAD_COMPETITION_RUNBOOK.md             # Training and submission runbook
└── TASK3_TASK4_SUBMISSION_EVALUATION.md       # Intermediate evaluation, superseded
```

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

A hybrid approach: a learned density model feeds the same analytic pencil-beam engine
used in Task 3. The two components are:

1. **Synthetic CT network** — a 3D residual U-Net (31.4 M parameters, 5 resolution levels,
   base width 24, channel widths 24→48→96→192→384, GroupNorm, SiLU activations, softplus
   output) maps single-channel MRI patches to synthetic CT in the encoded HU range [0, 1].
   Training uses a tissue-weighted L1 loss (bone ×3, lung ×2) with AdamW, 96³ patches
   biased to body interior, and cosine annealing over 40 epochs after a 7-epoch warm start
   on the same data. No external data or pre-trained weights.

2. **Pencil-beam engine** — identical to Task 3. Once the synthetic CT is generated per
   volume (sliding-window, 25% overlap, Gaussian blending, ~11 s on the evaluation GPU),
   beamlets are computed on CPU by a pool of single-threaded workers reading the density
   copy-on-write. Thread pinning via `threadpoolctl` is critical: without it, adding
   workers makes throughput worse, not better.

The key insight: MRI carries no density information, and the two tissues at the extremes of
the density scale — cortical bone (~1.8 g/cm³) and aerated lung (~0.3 g/cm³) — both appear
dark on bSSFP MRI. Treating the body as water works for abdominal patients (beamlet MAE
0.013–0.033) but fails badly in the thorax (0.18–0.29). Making density inference an explicit,
separately supervised subproblem — using the challenge's paired MRI–CT volumes — reduced IDD
curve distance from 0.168 (end-to-end dose network) to 0.024 in the preliminary phase.
Checkpoint selection used downstream beamlet-dose fidelity on held-out patients, not HU
reconstruction error.

The full method for both tasks is described in the LNCS-format report submitted to the
organizers.

### Task 1 and Task 2 — photon dose (development only)

Task 1 uses the same 3D residual U-Net architecture with 6 conditioning channels: CT volume,
body mask, MLC aperture projection, and divergent beam coordinates from SAD. It includes
radiological depth features for physics-informed conditioning. Task 2 warm-starts from the
Task 1 checkpoint and replaces CT with MRI input while keeping all other conditioning
channels identical. Neither container met the runtime limit on the evaluation platform.

## Reproducing the results

Challenge data is **not** included in this repository and is not redistributed here; it is
available from the DoseRAD2026 organizers under the challenge's own terms.

### Environment

```
Python      3.13
PyTorch     2.9.1 (CUDA 12.6, cuDNN 9)
pyRadPlan   0.3.5
NumPy       2.3.4
SimpleITK   2.5.6
FastAPI     0.141.1
```

### Installation

```bash
python3 -m venv .venv && source .venv/bin/activate

# Core dependencies (Task 3 and Task 4 submission)
pip install -r doserad_proton_ct/submission/requirements.txt

# Pencil-beam engine (pyRadPlan + threadpoolctl)
pip install -r doserad_proton_ct/submission/requirements-pb.txt

# For Task 1/2 training (torch, scipy, SimpleITK)
pip install -r doserad_photon_ct/requirements.txt
```

### Task 3 — build the pencil-beam submission container

Task 3 needs no trained weights: the engine reads density from the planning CT.
`build.sh` defaults to `DOSE_ENGINE=network`, so the pencil-beam image is built by
passing the build argument explicitly.

```bash
cd doserad_proton_ct

# Build the pencil-beam image (linux/amd64, on pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime)
docker build --platform=linux/amd64 \
  --build-arg TASK=proton-ct --build-arg DOSE_ENGINE=pencilbeam \
  --file submission/Dockerfile --tag doserad2026-proton-ct:latest ..

# Save the image as a tarball for Grand Challenge upload
mkdir -p dist && docker save doserad2026-proton-ct:latest | gzip -c > dist/doserad2026-proton-ct.tar.gz

# Run a local smoke test — starts the container, hits /health and /invoke.
# The script still gates on dist/proton-ct/best.pt, a leftover from the network
# backend; the pencil-beam backend itself needs no dose checkpoint.
bash submission/smoke_test.sh proton-ct
```

### Task 4 — train the synthetic-CT network, then build submission

The submitted Task 4 model is the MRI→synthetic-CT network trained by
`scripts/train_synthetic_ct.py`. The dose-prediction scripts in the same folder
(`scripts/train.py`, `scripts/train_all_gpu.sh`, `scripts/train_v*.sh`) belong to the
earlier end-to-end approach that this one replaced; they are kept for the record and
do not reproduce the submission.

```bash
cd doserad_proton_ct

# Prepare the dataset (audit, manifest, splits)
python3 scripts/prepare_dataset.py

# Warm-start run: 7 epochs at 2e-4
python3 scripts/train_synthetic_ct.py \
  --epochs 7 --learning-rate 2e-4 \
  --output-dir runs/synthetic_ct_warm

# Final run: 40 epochs at 1.5e-4 with cosine annealing, warm-started from the above
python3 scripts/train_synthetic_ct.py \
  --epochs 40 --learning-rate 1.5e-4 \
  --init-from runs/synthetic_ct_warm/best.pt \
  --output-dir runs/synthetic_ct

# Package the synthetic-CT checkpoint (no dose checkpoint for the pencil-beam backend)
bash submission/package_model.sh - dist/proton-mri/model.tar.gz runs/synthetic_ct/best.pt

# Build and test the submission container
docker build --platform=linux/amd64 \
  --build-arg TASK=proton-mri --build-arg DOSE_ENGINE=pencilbeam \
  --file submission/Dockerfile --tag doserad2026-proton-mri:latest ..
# The smoke script mounts dist/proton-mri at /opt/ml/model, where the backend
# picks up sct.pt; it also gates on a best.pt in that folder (network-backend leftover).
bash submission/smoke_test.sh proton-mri
```

Checkpoints were selected by downstream beamlet-dose fidelity rather than HU error;
`scripts/evaluate_checkpoint.py` runs that comparison against held-out patients.

### Task 1 (development only)

```bash
cd doserad_photon_ct

# One-time data preparation: audit, manifest, splits, normalization stats
bash scripts/finalize_dataset.sh

# Train on GPU (3D U-Net, AMP, gradient accumulation)
bash scripts/train_gpu.sh

# Evaluate full validation volumes
python3 scripts/evaluate_checkpoint.py \
  --checkpoint runs/photon_ct_baseline/best.pt \
  --device cuda --max-records 10 --batch-size 2

# Predict a single dose volume
python3 scripts/predict_volume.py \
  --checkpoint runs/photon_ct_baseline/best.pt \
  --patient-id 1ABB006 --beam-idx 0 --cp-idx 0 \
  --device cuda --output artifacts/example_prediction.mha

# Build submission container
bash submission/build.sh
bash submission/smoke_test.sh
```

### Running tests

```bash
# Task 1 tests (indexing, MHA I/O, conditioning, training, inference)
cd doserad_photon_ct && python3 -m unittest discover -s tests -v

# Task 3/4 tests (proton pipeline, submission output streaming)
cd doserad_proton_ct && python3 -m unittest discover -s tests -v
```

### Hardware

All training and development was done on a single NVIDIA DGX Spark (ARM64).
Submission containers were cross-built for x86 (linux/amd64) evaluation using
Docker buildx with QEMU emulation.

## Evaluation metrics

The official evaluation scores each submission on six axes, with runtime
double-weighted in the final ranking:

| Metric | Description |
| --- | --- |
| Beam MAE | Masked per-beam mean absolute error |
| IDD | Integrated depth-dose curve distance |
| Plan MAE | Plan-level stratified mean absolute error |
| Gamma 1%/1mm | Local 3D gamma pass rate |
| DVH | Dose-volume histogram clinical scores |
| Runtime | Wall-clock inference time (double-weighted) |

## Challenge

DoseRAD2026 concluded on 2 September 2026 with 46 active participants and 565 algorithm
submissions, 59 of them in the final test phase. Organized by the German Cancer Research
Center (DKFZ), the Paul Scherrer Institute, Delft University of Technology, Amsterdam
University Medical Center and University Medical Center Utrecht, among others.

### Official resources

- **Challenge:** <https://doserad2026.grand-challenge.org/>
- **Dataset:** <https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/DoseRAD2026>
- **Baseline:** <https://github.com/DoseRAD2026/pyradplan-pb-baseline>
- **Submission template:** <https://github.com/DoseRAD2026/example-submission>
- **Evaluation code:** <https://github.com/DoseRAD2026/evaluation-setup>

## License

MIT — see [LICENSE](LICENSE). This covers the code in this repository.

Two things it does not cover. `llncs.cls` in the `paper*/` folders is Springer's LNCS
document class, distributed under its own terms; the LNCS reports themselves are the
author's work but are made available here for reading rather than reuse. The challenge
data is not in this repository at all and remains under the DoseRAD2026 organizers' terms.
