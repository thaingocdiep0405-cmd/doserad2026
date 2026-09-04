# DoseRAD2026 Task 2 — Photon dose on MRI

Task 2 is isolated from the Task 1 project. It consumes only the MRI volume and
photon beam metadata at inference time. CT and CT-derived density are never read
by the model or submission adapter.

## Data and split

- Dataset: `data/photon/training` (relative to repo root)
- Complete MRI patients: 75
- Photon control points: 40,500
- Split: the same patient-level 60/15 split as Task 1
- MRI preprocessing: zero-background mask and per-volume foreground p1/p99 scaling

## Training

The main run warm-starts the compatible six-channel Task 1 checkpoint and
fine-tunes all weights on MRI:

```bash
cd doserad_photon_mri
bash scripts/train_gpu.sh 2>&1 | tee runs/photon_mri_baseline/train.log
```

Follow progress:

```bash
tail -f runs/photon_mri_baseline/train.log
```

Task 2 checkpoints and submission artifacts must remain under this project;
they are not interchangeable with the Task 1 container.
