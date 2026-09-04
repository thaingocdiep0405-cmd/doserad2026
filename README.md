# DoseRAD2026 — Radiation Dose Prediction Challenge

My solution for the [DoseRAD2026 Grand Challenge](https://doserad2026.grand-challenge.org/), covering all four tasks of 3D radiation dose prediction from medical imaging.

**Final submissions:** Task 3 (Proton CT) and Task 4 (Proton MRI) were submitted to the final evaluation phase. Task 1 (Photon CT) and Task 2 (Photon MRI) are included in this repository as complete training pipelines but were not submitted to the final phase.

## Challenge Overview

DoseRAD2026 challenges participants to predict per-control-point 3D radiation dose distributions from patient imaging (CT or MRI) and beam metadata. Each task targets a different modality and particle combination:

| Task | Directory | Modality | Particle | Status |
|------|-----------|----------|----------|--------|
| **Task 1** | `doserad_photon_ct/` | CT | Photon | Preliminary only |
| **Task 2** | `doserad_photon_mri/` | MRI | Photon | Preliminary only |
| **Task 3** | `doserad_proton_ct/` | CT | Proton | Final submitted |
| **Task 4** | `doserad_proton_mri/` | MRI | Proton | Final submitted |

## Architecture

All tasks share a common architecture: a **conditioned 3D U-Net** with non-negative residual blocks and dose-aware loss functions. Each task adapts the conditioning channels to its input modality:

- **Photon tasks (1 & 2):** CT/body mask, MLC aperture projection, beam-coordinate conditioning with divergent beam projection from SAD
- **Proton tasks (3 & 4):** Image intensity, body/density, ray geometry, Gaussian spot profile, energy spread and spatial coordinates

### Cross-task warm-starting

- Task 2 (Photon MRI) warm-starts from the Task 1 (Photon CT) checkpoint
- Task 4 (Proton MRI) warm-starts from the Task 3 (Proton CT) checkpoint

This transfer learning strategy leverages the shared beam physics between CT and MRI tasks within each particle type.

## Repository Structure

```
doserad2026/
├── README.md                          # This file
├── DOSERAD2026_MASTER_ROADMAP.md      # Development roadmap
├── DOSERAD_COMPETITION_RUNBOOK.md     # Monitoring and submission guide
├── .gitignore
│
├── doserad_photon_ct/                 # Task 1: Photon CT
│   ├── src/doserad_photon_ct/         # Core library
│   │   ├── model.py                   #   3D U-Net architecture
│   │   ├── dataset.py                 #   Patch-based data loader
│   │   ├── conditioning.py            #   Beam conditioning channels
│   │   ├── inference.py               #   Sliding-window inference
│   │   ├── losses.py                  #   Dose-aware loss functions
│   │   ├── metrics.py                 #   Evaluation metrics
│   │   ├── mha.py                     #   MHA reader/writer
│   │   ├── radiological.py            #   Radiological depth features
│   │   └── dataset_index.py           #   Manifest indexer
│   ├── scripts/                       # Training, evaluation, utility scripts
│   ├── submission/                    # Docker container for Grand Challenge
│   ├── paper/                         # LNCS paper
│   ├── tests/                         # Unit tests
│   └── artifacts/                     # Evaluation results
│
├── doserad_photon_mri/                # Task 2: Photon MRI
│   ├── src/doserad_photon_mri/        # Core library (MRI-adapted)
│   ├── scripts/
│   ├── submission/
│   ├── tests/
│   └── artifacts/
│
├── doserad_proton_ct/                 # Task 3 & 4: Proton CT + MRI
│   ├── src/doserad_proton/            # Shared proton library
│   │   ├── conditioning.py            #   Proton beam conditioning
│   │   ├── data.py                    #   Dataset for 81k pencil-beam maps
│   │   ├── inference.py               #   Inference engine
│   │   └── pencilbeam.py              #   Pencil-beam dose engine
│   ├── scripts/
│   ├── submission/
│   ├── paper/                         # Task 3 paper
│   ├── paper_t3/                      # Task 3 LNCS paper
│   ├── tests/
│   └── artifacts/
│
├── doserad_proton_mri/                # Task 4: Proton MRI (uses proton_ct code)
│   └── artifacts/
│
├── run_*.sh                           # Orchestration scripts
└── data/                              # Dataset (not tracked, see below)
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.5 (CUDA)
- NumPy >= 2.0
- SciPy >= 1.13
- SimpleITK >= 2.4

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r doserad_photon_ct/requirements.txt
```

## Dataset

The dataset is hosted on Hugging Face and not included in this repository:

- **Download:** https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/DoseRAD2026
- **Location:** Place downloaded data under `data/photon/training` and `data/proton/training`
- **Structure:** Each patient contains a CT/MRI volume, beam geometry JSON, and per-control-point Monte Carlo dose maps

### Data preparation

After downloading, run the finalization script for each task:

```bash
# Task 1: Photon CT
cd doserad_photon_ct
bash scripts/finalize_dataset.sh

# Task 2: Photon MRI
cd doserad_photon_mri
bash scripts/finalize_dataset.sh
```

This audits headers, rebuilds the manifest and splits, and computes normalization statistics. The split is deterministic, patient-level, and anatomy-stratified (60 train / 15 validation).

## Training

### Task 1 — Photon CT

```bash
cd doserad_photon_ct
bash scripts/train_gpu.sh
```

### Task 2 — Photon MRI

Requires a trained Task 1 checkpoint for warm-starting:

```bash
cd doserad_photon_mri
bash scripts/train_gpu.sh
```

### Task 3 — Proton CT

```bash
cd doserad_proton_ct
bash scripts/train_all_gpu.sh
```

### Task 4 — Proton MRI

Automatically queued after Task 3 by `train_all_gpu.sh`, or run separately:

```bash
cd doserad_proton_ct
bash scripts/train_v5_mri.sh
```

### Orchestrated training

Run multiple tasks sequentially on a single GPU:

```bash
bash run_doserad_tasks_1_2.sh    # Task 1 then Task 2
bash run_v5_training_queue.sh    # Queued training runs
```

## Evaluation

Evaluate a checkpoint against validation patients:

```bash
python3 scripts/evaluate_checkpoint.py \
  --checkpoint runs/<run_name>/best.pt \
  --device cuda \
  --max-records 10 \
  --batch-size 2
```

### Metrics

The official evaluation includes:
- **Beam MAE** — masked per-beam mean absolute error
- **IDD** — integrated depth-dose distance
- **Plan MAE** — plan-level stratified mean absolute error
- **Gamma 1%/1mm** — local 3D gamma pass rate
- **DVH scores** — dose-volume histogram clinical metrics
- **Runtime** — inference speed (double-weighted in ranking)

## Submission

Each task produces a Docker container for Grand Challenge evaluation.

### Build a submission container

```bash
cd doserad_photon_ct

# Package model weights
bash submission/package_model.sh runs/<run_name>/best.pt dist/model.tar.gz

# Build Docker image (linux/amd64)
bash submission/build.sh

# Save image
bash submission/save_image.sh dist/doserad2026-photon-ct.tar.gz
```

### Submission contract

The container must:
- Serve a `/health` endpoint returning `200`
- Accept `/invoke` requests returning `201`
- Load model weights once at startup
- Accept up to 10 stacked input volumes with beam metadata
- Write 10 stacked radiation-dose `.mha` outputs
- Include the Docker label `org.grand-challenge.api-method="invoke"`

### Local smoke test

```bash
bash submission/smoke_test.sh
```

## Key Design Decisions

- **Patch-based training** with positive-dose-biased sampling to handle sparse dose distributions
- **Sliding-window inference** with Gaussian blending for seamless full-volume predictions
- **Non-negative residual U-Net** to enforce physical dose positivity
- **Fixed absolute dose scale** (`1e-4`) — never per-sample normalization, as absolute dose is part of the task
- **Leakage-safe splits** — all control points of a patient stay in the same split
- **Pencil-beam physics engine** for proton tasks with batch processing for runtime optimization

## Testing

```bash
# Run all tests for a task
cd doserad_photon_ct
python3 -m unittest discover -s tests -v
```

Tests cover indexing, MHA I/O, conditioning, dataset loading, forward/backward training, and sliding inference.

## Official Resources

- **Challenge:** https://doserad2026.grand-challenge.org/
- **Dataset:** https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/DoseRAD2026
- **Baseline:** https://github.com/DoseRAD2026/pyradplan-pb-baseline
- **Submission template:** https://github.com/DoseRAD2026/example-submission
- **Evaluation code:** https://github.com/DoseRAD2026/evaluation-setup

## License

This project is developed for the DoseRAD2026 Grand Challenge. Please refer to the challenge rules for usage terms.
