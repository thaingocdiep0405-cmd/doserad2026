# Photon MRI optimization report

Updated: 2026-08-14

## Selected candidate

- checkpoint: `runs/photon_mri_baseline/best_idd.pt`;
- inference patch: 128 x 128 x 128, overlap 0.25;
- CUDA batch: 8 control points;
- outer chunk: 64 control points;
- AMP FP16 kernels with FP32 external input;
- `torch.compile(mode="reduce-overhead", fullgraph=True)`;
- model warm-up during `load_model` and padded partial inference batches;
- empty-aperture pruning and zero dose outside the MRI foreground.

Checkpoint blends were rejected because they improved MAE but worsened IDD and
did not dominate `best_idd.pt` on the fixed validation cohort.

## Accuracy regression on 75 fixed validation records

| Runtime mode | Beam MAE | NRMSE | IDD |
| --- | ---: | ---: | ---: |
| eager | 0.085325 | 0.018418 | 0.309045 |
| compiled + padded | 0.085348 | 0.018428 | 0.309436 |

The compiled numerical path changes MAE by 0.026%, NRMSE by 0.053%, and IDD by
0.127%.

## Runtime benchmark

On the local NVIDIA GB10, 64 control points from patient `1ABB039` take 35.26
seconds with the selected compiled configuration. The linear 181-control-point
projection is 99.73 seconds. Peak CUDA memory is 6.33 GiB allocated and 10.92
GiB reserved. The prior realistic eager chunk-16 path projected 205.53 seconds.

These are local engineering measurements, not the authoritative challenge
runtime. The final amd64 container must still be measured on the official A10G.
