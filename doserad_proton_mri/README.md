# DoseRAD2026 Proton MRI (Task 4)

Task 4 uses the shared implementation in `../doserad_proton_ct` and writes its
own checkpoints under `runs/proton_mri_v1_teacher_student`. Training is queued
after Task 3 by `../doserad_proton_ct/scripts/train_all_gpu.sh` and warm-starts
from the best Proton CT checkpoint. CT is used only as the source of initial
weights; Proton MRI samples and inference channels contain MRI, not CT.
