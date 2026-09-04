#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark batched proton inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--modality", choices=("ct", "mri"), required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts/manifest.csv")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "artifacts/splits.json")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--beamlets", type=int, default=16)
    parser.add_argument("--condition-batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--pad-batch", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--target-beamlets", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PhotonDoseUNet3D(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    if args.torch_compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    allowed = load_split_patients(args.splits, args.split)
    records = [record for record in read_manifest(args.manifest) if record.patient_id in allowed]
    patient_id = sorted({record.patient_id for record in records})[0]
    records = [record for record in records if record.patient_id == patient_id]
    base_conditions = [record.condition for record in records]
    if args.beamlets <= len(base_conditions):
        indices = np.linspace(
            0, len(base_conditions), num=args.beamlets, endpoint=False, dtype=np.int64
        )
        conditions = [base_conditions[int(index)] for index in indices]
    else:
        conditions = [
            base_conditions[index % len(base_conditions)] for index in range(args.beamlets)
        ]
    image_path = records[0].ct_path if args.modality == "ct" else records[0].mr_path
    image = np.asarray(load_mha_array(image_path), dtype=np.float32)
    geometry = SpatialGeometry.from_mha_header(read_mha_header(image_path))
    training_config = checkpoint["training_config"]
    gate = max(float(training_config.get("ray_gate_threshold", 0.0)), 1.0e-6)

    kwargs = dict(
        model=model,
        image=image,
        geometry=geometry,
        conditions=conditions,
        modality=args.modality,
        device=device,
        patch_size_zyx=tuple(args.patch_size),
        dose_scale=float(training_config["dose_scale"]),
        overlap=args.overlap,
        condition_batch_size=args.condition_batch_size,
        amp=True,
        skip_empty_ray=True,
        mask_outside_body=True,
        relative_cutoff=0.0,
        ray_gate_threshold=gate,
        pad_to_batch_size=args.pad_batch,
    )

    if args.warmup:
        warm_conditions = kwargs["conditions"]
        kwargs["conditions"] = conditions[: args.condition_batch_size]
        predict_conditioned_arrays(**kwargs)
        kwargs["conditions"] = warm_conditions

    durations: list[float] = []
    outputs = []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        outputs = predict_conditioned_arrays(**kwargs)
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)

    seconds = statistics.median(durations)
    payload = {
        "patient_id": patient_id,
        "modality": args.modality,
        "image_shape_zyx": list(image.shape),
        "beamlets": args.beamlets,
        "condition_selection": "evenly_spaced",
        "patch_size_zyx": list(args.patch_size),
        "overlap": args.overlap,
        "condition_batch_size": args.condition_batch_size,
        "torch_compile": args.torch_compile,
        "pad_batch": args.pad_batch,
        "warmup": args.warmup,
        "durations": durations,
        "seconds": seconds,
        "seconds_per_beamlet": seconds / args.beamlets,
        "estimated_seconds_for_target_beamlets": seconds / args.beamlets * args.target_beamlets,
        "target_beamlets": args.target_beamlets,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "output_count": len(outputs),
        "finite_outputs": all(np.isfinite(output).all() for output in outputs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
