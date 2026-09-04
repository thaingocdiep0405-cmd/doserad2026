# DoseRAD2026 Proton CT (Task 3)

Shared training pipeline for Task 3 (Proton CT) and Task 4 (Proton MRI). It
indexes all 81,000 pencil-beam dose maps and preserves the same leakage-safe
60/15 patient split used by the Photon experiments.

The ten input channels describe image intensity, body/density, ray geometry,
Gaussian spot profile, energy spread and spatial coordinates. Proton MRI uses
the same contract so it can warm-start from Proton CT without consuming CT at
MRI inference time.

Run `python3 scripts/prepare_dataset.py`, then use `scripts/train_all_gpu.sh` to
train CT followed by MRI on a single GPU.
