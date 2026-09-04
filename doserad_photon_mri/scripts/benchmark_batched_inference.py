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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from doserad_photon_mri.conditioning import SpatialGeometry  # noqa: E402
from doserad_photon_mri.dataset import (  # noqa: E402
    load_split_patients,
    lookup_condition,
    read_manifest,
)
from doserad_photon_mri.dataset_index import read_mha_header  # noqa: E402
from doserad_photon_mri.inference import (  # noqa: E402
    prepare_conditioned_inference,
    predict_conditioned_arrays,
    warmup_model,
)
from doserad_photon_mri.mha import load_mha_array  # noqa: E402
from doserad_photon_mri.metrics import (  # noqa: E402
    beam_direction,
    idd_curve_distance,
    masked_beam_mae,
)
from doserad_photon_mri.model import ModelConfig, PhotonDoseUNet3D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multi-control-point inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts/manifest.csv")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "artifacts/splits.json")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--control-points", type=int, default=16)
    parser.add_argument("--condition-batch-size", type=int, default=4)
    parser.add_argument(
        "--condition-chunk-size",
        type=int,
        default=0,
        help="Outer submission chunk size; 0 predicts all control points in one call",
    )
    parser.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--model-warmup", action="store_true")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--pad-batch", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/batched_benchmark.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if args.warmup_runs < 0 or args.repeats < 1:
        raise ValueError("warmup-runs must be non-negative and repeats must be positive")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PhotonDoseUNet3D(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    if args.torch_compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    config = checkpoint["training_config"]
    include_physics_priors = bool(
        checkpoint["model_config"].get("physics_priors", False)
    )

    allowed = load_split_patients(args.splits, args.split)
    grouped = {}
    for record in read_manifest(args.manifest):
        if record.patient_id in allowed:
            grouped.setdefault(record.patient_id, []).append(record)
    patient_id = sorted(grouped)[0]
    records = sorted(
        grouped[patient_id], key=lambda item: (item.beam_idx, item.cp_idx)
    )[: args.control_points]
    if len(records) < args.control_points:
        raise ValueError(f"patient {patient_id} only has {len(records)} control points")

    image = np.asarray(load_mha_array(records[0].image_path), dtype=np.float32)
    geometry = SpatialGeometry.from_mha_header(read_mha_header(records[0].image_path))
    conditions = [lookup_condition(record) for record in records]
    if args.model_warmup:
        warmup_model(
            model,
            device=device,
            batch_size=args.condition_batch_size,
            in_channels=int(checkpoint["model_config"]["in_channels"]),
            patch_size_zyx=tuple(args.patch_size),
            amp=True,
        )
    inference_kwargs = dict(
        model=model, image=image, geometry=geometry, conditions=conditions,
        device=device, patch_size_zyx=tuple(args.patch_size),
        dose_scale=float(config["dose_scale"]), overlap=args.overlap,
        condition_batch_size=args.condition_batch_size, amp=True,
        include_physics_priors=include_physics_priors,
        skip_empty_aperture=True, mask_outside_body=True,
        pad_to_batch_size=args.pad_batch,
    )

    def predict_all() -> list[np.ndarray]:
        prepared = None
        if args.reuse_prepared:
            prepared = prepare_conditioned_inference(
                image,
                geometry,
                device=device,
                patch_size_zyx=tuple(args.patch_size),
                overlap=args.overlap,
            )
        if args.condition_chunk_size <= 0:
            return predict_conditioned_arrays(**inference_kwargs, prepared=prepared)
        outputs = []
        for offset in range(0, len(conditions), args.condition_chunk_size):
            chunk_kwargs = dict(inference_kwargs)
            chunk_kwargs["conditions"] = conditions[
                offset : offset + args.condition_chunk_size
            ]
            outputs.extend(
                predict_conditioned_arrays(**chunk_kwargs, prepared=prepared)
            )
        return outputs

    for _ in range(args.warmup_runs):
        predict_all()
    durations = []
    predictions = []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.repeats):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        predictions = predict_all()
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
    seconds = statistics.median(durations)

    maes = []
    idds = []
    for record, condition, prediction in zip(records, conditions, predictions):
        target = np.asarray(load_mha_array(record.dose_path), dtype=np.float32)
        maes.append(masked_beam_mae(prediction, target))
        idds.append(
            idd_curve_distance(
                prediction,
                target,
                beam_direction(condition.gantry_angle_deg),
                geometry.spacing_xyz,
            )
        )
    payload = {
        "patient_id": patient_id,
        "control_points": len(records),
        "mri_shape_zyx": list(image.shape),
        "patch_size_zyx": list(args.patch_size),
        "overlap": args.overlap,
        "condition_batch_size": args.condition_batch_size,
        "condition_chunk_size": args.condition_chunk_size,
        "warmup_runs": args.warmup_runs,
        "model_warmup": args.model_warmup,
        "reuse_prepared": args.reuse_prepared,
        "torch_compile": args.torch_compile,
        "pad_batch": args.pad_batch,
        "repeats": args.repeats,
        "durations": durations,
        "seconds": seconds,
        "seconds_per_control_point": seconds / len(records),
        "estimated_seconds_for_181_control_points": seconds / len(records) * 181,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "mean_masked_beam_mae": float(np.mean(maes)),
        "mean_idd_curve_distance": float(np.mean(idds)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
