# DoseRAD2026 — master roadmap and quality gates

This workspace targets all four challenge variants with separate models and
submission containers. A result is considered ready only after every required
gate below passes; a decreasing training loss alone is not sufficient.

## Official tasks

| Task | Image | Beam | Project | Current state |
|---|---|---|---|---|
| 1 | CT | Photon VMAT control point | `doserad_photon_ct` | trained, locally evaluated and container smoke-tested |
| 2 | MRI | Photon VMAT control point | `doserad_photon_mri` | trained, validated and container-contract smoke-tested; A10G test pending |
| 3 | CT | Proton pencil beamlet | `doserad_proton_ct` | blocked by incomplete public download |
| 4 | MRI | Proton pencil beamlet | `doserad_proton_mri` | blocked by incomplete public download and Task 3 |

Task inputs at inference are strictly modality-specific. A CT task cannot rely
on MRI and an MRI task cannot rely on CT because the hidden-test container only
receives the selected image modality.

## Model strategy

### Photon CT

Residual 3D U-Net conditioned on CT, body mask, MLC aperture, beam depth,
lateral coordinate and superior-inferior coordinate. The optimized Task 1 model
also uses CT-derived mass density.

### Photon MRI

Six-channel residual 3D U-Net initialized from the compatible Task 1 weights.
It uses robust per-volume MRI foreground normalization and never reads CT at
inference. A later experiment may add an MRI-to-density auxiliary head, but it
must be selected on held-out patients rather than assumed to improve accuracy.

### Proton CT

Separate energy-conditioned 3D model. Required conditioning includes ray source,
ray target/direction, beamlet energy, transverse distance to the ray and CT-based
mass/stopping-power information. Depth and water-equivalent path length are
critical for learning the Bragg peak and must be validated with IDD distance.

### Proton MRI

The Proton CT beam model is reused, while the material representation is inferred
from MRI. Paired training CT may be used as an auxiliary supervision signal, but
the final inference graph receives MRI only.

## Mandatory data gates

1. Exactly 75 complete training patients for the selected beam modality.
2. Photon: 40,500 expected/available dose maps.
3. Proton: 81,000 expected/available dose maps.
4. Image, label and metadata geometry match.
5. Patient-level stratified split; no control points/beamlets from one patient
   may occur in both train and validation.
6. Fixed validation cohort, seeds and sampled beam list are versioned.

## Mandatory accuracy gates

Report validation metrics as patient-level aggregates, matching the official
ranking semantics:

- beam masked MAE (lower is better),
- IDD curve distance (lower is better),
- stratified plan MAE (lower is better),
- local 3D gamma 1%/1 mm (higher is better),
- DVH clinical score (lower is better),
- mean, standard deviation and anatomy-stratified results.

Model selection uses held-out patients and full-volume predictions. Checkpoints
are selected separately for important metrics when the optima differ, followed
by a predeclared balanced selection or checkpoint blend. Test/leaderboard scores
must never be used repeatedly as a hyperparameter search set.

## Mandatory engineering gates

Each of the four containers must independently pass:

1. amd64 build and model load without network access,
2. `GET /health` returns 200 after model loading,
3. `POST /invoke` returns 201,
4. input directory is `source-ct-image-*` or `source-mri-image-*` as appropriate,
5. all ten output slots exist and each dose stack is genuine scalar 4D,
6. output geometry exactly matches its input image,
7. placement follows `output_file_idx` and `idx_in_output`,
8. every value `<= minimum_cutoff` is set to zero,
9. no NaN/Inf and no negative dose,
10. runtime measured on the official NVIDIA A10G 24 GiB hardware.

Photon runtime must be below 181 seconds for the representative 181-map case;
proton runtime must be below 500 seconds for the representative 500-map case.
A safety target of at most 80% of the hard limit is used before final submission.

## Submission discipline

- Use preliminary submissions first; at most 10 are allowed per task.
- Reserve final submissions for two frozen candidates per task.
- Do not submit while an organizer warning says a scoring metric is being fixed.
- Record the container digest, checkpoint checksum, configuration, local metrics,
  runtime and submission ID for every submitted candidate.
- A leaderboard rank is evidence about the hidden test set, not proof of clinical
  suitability. Challenge outputs are research results and require independent
  clinical validation before any patient use.

## Immediate execution order

1. Package and smoke-test the optimized Photon CT and Photon MRI containers.
2. Measure both frozen containers on an NVIDIA A10G 24 GiB fixture.
3. Allow the ongoing proton download to reach 75 patients / 81,000 maps.
4. Build and train Proton CT, with IDD/WEPL checks from the first baseline.
5. Transfer the proton beam branch to MRI and train Proton MRI.
6. Submit frozen candidates only after the organizer's current IDD warning is gone.
