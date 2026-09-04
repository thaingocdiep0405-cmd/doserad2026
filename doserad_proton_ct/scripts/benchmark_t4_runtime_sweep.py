#!/usr/bin/env python3
"""Uncontended runtime sweep for the Task 4 (proton MRI) submission config.

Replays the submission chunk loop (window cache shared across chunks) for 64
real validation beamlets under several patch/mode/batch/chunk combinations and
reports seconds per map plus the projected hidden runtime (local GB10 to A10G
factor measured from the T3/T4 preliminary feedback).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE / "doserad_photon_ct" / "src"))

from doserad_photon_ct.conditioning import SpatialGeometry  # noqa: E402
from doserad_photon_ct.dataset_index import read_mha_header  # noqa: E402
from doserad_photon_ct.mha import load_mha_array  # noqa: E402
from doserad_photon_ct.model import ModelConfig, PhotonDoseUNet3D  # noqa: E402
from doserad_proton.data import load_split_patients, read_manifest  # noqa: E402
from doserad_proton.inference import predict_conditioned_arrays  # noqa: E402

GB10_TO_A10G = 2.07


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--modality", choices=("ct", "mri"), default="mri")
    parser.add_argument("--beamlets", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


CONFIGS = [
    # (label, patch, roi_mode, batch, chunk)
    ("submitted: p96 corridor b8 c32", (96, 96, 96), "corridor", 8, 32),
    ("p96 capsule b8 c32", (96, 96, 96), "capsule", 8, 32),
    ("p96 capsule b16 c64", (96, 96, 96), "capsule", 16, 64),
    ("p96 capsule b32 c64", (96, 96, 96), "capsule", 32, 64),
    ("p128 capsule b16 c64", (128, 128, 128), "capsule", 16, 64),
    ("p128 corridor b16 c64", (128, 128, 128), "corridor", 16, 64),
]


def main() -> int:
    args = parse_args()
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PhotonDoseUNet3D(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dose_scale = float(checkpoint["training_config"]["dose_scale"])

    patients = load_split_patients(PROJECT_ROOT / "artifacts/splits.json", "validation")
    grouped = defaultdict(list)
    for record in read_manifest(PROJECT_ROOT / "artifacts/manifest.csv"):
        if record.patient_id in patients:
            grouped[record.patient_id].append(record)
    patient_id = sorted(grouped)[0]
    records = sorted(
        grouped[patient_id], key=lambda r: (r.beam_idx, r.ray_idx, r.beamlet_idx)
    )
    step = max(1, len(records) // args.beamlets)
    records = records[::step][: args.beamlets]
    image_path = records[0].ct_path if args.modality == "ct" else records[0].mr_path
    image = np.asarray(load_mha_array(image_path), dtype=np.float32)
    geometry = SpatialGeometry.from_mha_header(read_mha_header(image_path))
    conditions = [record.condition for record in records]
    print(
        f"patient={patient_id} beamlets={len(conditions)} shape={image.shape}",
        flush=True,
    )

    results = []
    reference = None
    for label, patch, mode, batch, chunk in CONFIGS:
        window_cache: dict = {}
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = []
        for offset in range(0, len(conditions), chunk):
            outputs += predict_conditioned_arrays(
                model,
                image=image,
                geometry=geometry,
                conditions=conditions[offset : offset + chunk],
                modality=args.modality,
                device=device,
                patch_size_zyx=patch,
                dose_scale=dose_scale,
                overlap=0.0,
                condition_batch_size=batch,
                amp=True,
                skip_empty_ray=True,
                mask_outside_body=True,
                relative_cutoff=0.0,
                ray_gate_threshold=1e-4,
                pad_to_batch_size=True,
                roi_mode=mode,
                window_cache=window_cache,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        per_map = elapsed / len(conditions)
        hidden_500 = per_map * 500 * GB10_TO_A10G
        if reference is None:
            reference = outputs
            delta = 0.0
        else:
            delta = max(
                float(np.max(np.abs(a - b))) for a, b in zip(outputs, reference)
            )
        row = {
            "label": label,
            "seconds_64": round(elapsed, 2),
            "ms_per_map": round(per_map * 1000, 1),
            "hidden_est_500maps_s": round(hidden_500, 1),
            "max_voxel_delta_vs_first": delta,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        del outputs
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
