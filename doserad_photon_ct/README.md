# DoseRAD2026 photon-CT training pipeline

This directory contains an end-to-end neural baseline for the DoseRAD2026
`photon-ct` task. It audits the downloaded data, creates leakage-safe splits,
trains a conditioned 3D U-Net, evaluates full control-point dose volumes, and
packages a Grand Challenge inference container. It never modifies source
patient files under `../data`.

## Implemented components

- integrity audit and one-row-per-control-point manifest;
- deterministic patient-level, anatomy-stratified split;
- compressed MHA reader/writer with geometry preservation;
- CT/body/MLC aperture/beam-coordinate conditioning, including divergent beam
  projection from SAD;
- positive-dose-biased 3D patch dataset with per-worker random sampling and CT
  cache;
- non-negative residual 3D U-Net, dose-aware loss, AMP, gradient accumulation,
  checkpointing and resume;
- sliding-window full-volume inference with Gaussian blending;
- local full-volume validation and one-volume prediction CLI;
- Grand Challenge `/health` and `/invoke` adapter for stacked photon metadata,
  ten output stacks, cutoff handling, amd64 CUDA image build and model packaging;
- automated unit tests covering indexing, MHA I/O, conditioning, dataset,
  forward/backward training and sliding inference.

## What one training sample contains

For each patient:

- `image/ct.mha`: source CT volume.
- `<patient_id>.json`: beam geometry and VMAT control points.
- `dose/Dose_B<beam>_CP<control-point>.mha`: Monte Carlo target dose map.

The JSON currently contains three beams with 180 control points each for the
inspected sample, giving 540 target dose maps for that patient. The manifest
stores one row per control point while the MLC arrays remain in the JSON.

## Install and verify

Use a Python environment with PyTorch, NumPy and SimpleITK. Training itself can
read the dataset without SimpleITK; SimpleITK is needed by the submission
adapter and official tooling.

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

Check whether the full dataset and optional official baseline dependencies are
available:

```bash
python3 scripts/check_readiness.py
```

## Finalize the data

Do this once the Hugging Face download has completely stopped:

```bash
bash scripts/finalize_dataset.sh
```

This runs the header audit, rebuilds the manifest and split, computes CT
statistics over all 60 training patients plus five sampled dose maps per
training patient, and prints readiness. Validation patients are deliberately
excluded from normalization statistics. While downloading,
`audit_dataset.py` excludes incomplete patients; therefore artifacts generated
before completion are temporary.

For a quick inspection:

```bash
python3 scripts/inspect_sample.py --row 0
```

The split is deterministic, patient-level, and stratified by the anatomical
group encoded in patient IDs. Keeping all control points of a patient in one
split prevents leakage between train and validation.

## Train on GPU

The default launch configuration is a conservative starting point for one GPU:

```bash
bash scripts/train_gpu.sh
```

Equivalent direct command:

```bash
python3 scripts/train.py \
  --device cuda \
  --output-dir runs/photon_ct_baseline \
  --epochs 100 --steps-per-epoch 1000 --val-steps 100 \
  --batch-size 4 --gradient-accumulation 1 \
  --patch-size 96 96 96 --base-channels 12 \
  --num-workers 4 --full-val-every 5 --full-val-samples 2
```

Checkpoints are written atomically to `runs/photon_ct_baseline/last.pt` and
`best.pt`; metrics are appended to `metrics.jsonl`. Resume with the same model
and preprocessing settings:

```bash
python3 scripts/train.py \
  --device cuda \
  --resume runs/photon_ct_baseline/last.pt \
  --output-dir runs/photon_ct_baseline \
  --epochs 100 --steps-per-epoch 1000 --val-steps 100 \
  --batch-size 4 --gradient-accumulation 1 \
  --patch-size 96 96 96 --base-channels 12 \
  --num-workers 4 --full-val-every 5 --full-val-samples 2 \
  --inference-overlap 0.25
```

All options from the original run must be repeated when resuming. Use
`runs/photon_ct_baseline/config.json` as the source of truth. If GPU memory is
insufficient, first reduce `--patch-size` to `80 80 80`, then reduce
`--base-channels`; if memory remains, increase `--inference-batch-size` only
after measuring it.

The default fixed dose scale is `1e-4`, based on the current sample. Re-run the
full data statistics before final training and change `--dose-scale` only if
the complete distribution materially differs. Never normalize each target by
its own maximum because absolute dose scale is part of the task.

## Evaluate and inspect predictions

Evaluate full validation volumes (slow but representative):

```bash
python3 scripts/evaluate_checkpoint.py \
  --checkpoint runs/photon_ct_baseline/best.pt \
  --device cuda --max-records 10 --batch-size 2
```

Write one prediction on the exact source grid:

```bash
python3 scripts/predict_volume.py \
  --checkpoint runs/photon_ct_baseline/best.pt \
  --patient-id 1ABB006 --beam-idx 0 --cp-idx 0 \
  --device cuda --output artifacts/example_prediction.mha --compress
```

The local evaluator reports masked beam MAE and normalized RMSE. The official
repository under `official/evaluation-setup` remains authoritative for final
IDD, plan MAE, gamma and DVH scoring.

## Build the submission

Package weights separately, as required by Grand Challenge:

```bash
bash submission/package_model.sh \
  runs/photon_ct_baseline/best.pt dist/model.tar.gz
```

Build and save the amd64 container image:

```bash
bash submission/build.sh
bash submission/save_image.sh dist/doserad2026-photon-ct.tar.gz
```

The development host is `aarch64`, while the official runtime is
`linux/amd64`. The final image therefore needs Docker buildx/QEMU locally or,
preferably, an amd64 CUDA build machine. A successful unit test does not replace
an end-to-end `/invoke` run using official test fixtures and the trained model.

Do not duplicate the CT volume 540 times during preprocessing. Cache one CT per
patient and load control-point dose targets on demand. CT background is `-1024`
HU in the inspected data, while individual control-point doses are sparse and
much smaller in magnitude than a composed plan dose. Compute dataset-wide
statistics before choosing fixed clipping or target-normalisation constants,
and always convert predictions back to absolute dose units for evaluation.

The official evaluator includes masked per-beam MAE and integrated depth-dose
distance, plus plan-level stratified MAE, local 3D gamma 1%/1 mm, and DVH-based
clinical scores. Accuracy and runtime therefore both need to be measured from
the first baseline.

Use a dedicated Python environment. GPU visibility must be verified inside that
environment with `torch.cuda.is_available()` before launching a long run.

## Submission contract

The official example submission receives up to ten CT volumes plus stacked
photon beam metadata. It must write ten stacked radiation-dose `.mha` outputs,
including valid placeholders for unused slots. The container keeps a server
alive, returns `200` from `/health`, returns `201` from `/invoke`, loads weights
once at startup, and carries the required
`org.grand-challenge.api-method="invoke"` Docker label.

Official resources:

- Challenge: https://doserad2026.grand-challenge.org/
- Dataset: https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/DoseRAD2026
- Baseline: https://github.com/DoseRAD2026/pyradplan-pb-baseline
- Submission template: https://github.com/DoseRAD2026/example-submission
- Evaluation code: https://github.com/DoseRAD2026/evaluation-setup
