#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from doserad_photon_ct.dataset import (  # noqa: E402
    ManifestRecord,
    load_split_patients,
    read_manifest,
)
from doserad_photon_ct.inference import predict_record_volume, warmup_model  # noqa: E402
from doserad_photon_ct.mha import load_mha_array  # noqa: E402
from doserad_photon_ct.metrics import (  # noqa: E402
    beam_direction,
    idd_curve_distance,
    masked_beam_mae,
)
from doserad_photon_ct.model import ModelConfig, PhotonDoseUNet3D  # noqa: E402
from doserad_photon_ct.dataset import lookup_condition  # noqa: E402
from doserad_photon_ct.dataset_index import read_mha_header  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full photon-CT control-point volumes from a checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "artifacts/manifest.csv"
    )
    parser.add_argument(
        "--splits", type=Path, default=PROJECT_ROOT / "artifacts/splits.json"
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--max-records",
        type=int,
        default=75,
        help="Maximum records, allocated as evenly as possible across patients and beams",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts/checkpoint_evaluation.json"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--patch-size", type=int, nargs=3, metavar=("Z", "Y", "X"))
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-empty-aperture", action="store_true")
    parser.add_argument("--mask-outside-body", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--pad-batch", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def select_records(
    records: list[ManifestRecord], patient_ids: set[str], maximum: int, seed: int
) -> list[ManifestRecord]:
    if maximum <= 0:
        raise ValueError("max-records must be positive")
    grouped: dict[str, list[ManifestRecord]] = {}
    for record in records:
        if record.patient_id in patient_ids:
            grouped.setdefault(record.patient_id, []).append(record)
    if not grouped:
        raise ValueError("the requested split has no manifest records")

    rng = random.Random(seed)
    patients = sorted(grouped)
    rng.shuffle(patients)
    maximum = min(maximum, sum(len(items) for items in grouped.values()))
    if maximum < len(patients):
        patients = patients[:maximum]

    base, remainder = divmod(maximum, len(patients))
    selected: list[ManifestRecord] = []
    for patient_index, patient_id in enumerate(patients):
        patient_limit = min(
            len(grouped[patient_id]), base + int(patient_index < remainder)
        )
        records_by_beam: dict[int, list[ManifestRecord]] = defaultdict(list)
        for record in grouped[patient_id]:
            records_by_beam[record.beam_idx].append(record)
        beam_ids = sorted(records_by_beam)
        beam_base, beam_remainder = divmod(patient_limit, len(beam_ids))
        for beam_index, beam_id in enumerate(beam_ids):
            beam_records = sorted(records_by_beam[beam_id], key=lambda item: item.cp_idx)
            beam_limit = min(
                len(beam_records), beam_base + int(beam_index < beam_remainder)
            )
            if beam_limit == 0:
                continue
            # Quantiles cover the whole arc instead of clustering around adjacent CPs.
            positions = np.floor(
                (np.arange(beam_limit, dtype=np.float64) + 0.5)
                * len(beam_records)
                / beam_limit
            ).astype(int)
            selected.extend(beam_records[int(position)] for position in positions)
    return selected


def aggregate_by_patient(details: list[dict]) -> list[dict]:
    """Average beam metrics within each patient, matching official Level-1 ranking."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in details:
        grouped[item["patient_id"]].append(item)

    result = []
    for patient_id, items in sorted(grouped.items()):
        patient = {"patient_id": patient_id, "records": len(items)}
        for key in ("masked_beam_mae", "normalized_rmse", "idd_curve_distance"):
            values = [item[key] for item in items if item[key] is not None]
            patient[key] = float(np.mean(values)) if values else None
        result.append(patient)
    return result


def bootstrap_confidence_interval(
    patient_metrics: list[dict],
    key: str,
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict | None:
    values = np.asarray(
        [item[key] for item in patient_metrics if item[key] is not None],
        dtype=np.float64,
    )
    if values.size == 0:
        return None
    if samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence-level must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "lower": float(np.quantile(means, tail)),
        "upper": float(np.quantile(means, 1.0 - tail)),
        "confidence_level": confidence_level,
        "bootstrap_samples": samples,
    }


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PhotonDoseUNet3D(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    training_config = checkpoint["training_config"]
    patch_size = tuple(int(value) for value in (args.patch_size or training_config["patch_size"]))
    dose_scale = float(training_config["dose_scale"])
    ct_clip = tuple(float(value) for value in training_config["ct_clip"])
    include_physics_priors = bool(
        checkpoint["model_config"].get("physics_priors", False)
    )
    include_radiological_depth = bool(
        checkpoint["model_config"].get("radiological_depth", False)
    )
    include_density = (
        int(checkpoint["model_config"].get("in_channels", 6)) >= 7
    )
    if args.torch_compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    if device.type == "cuda" and args.torch_compile:
        warmup_model(
            model,
            device=device,
            batch_size=args.batch_size,
            in_channels=int(checkpoint["model_config"]["in_channels"]),
            patch_size_zyx=patch_size,
            amp=not args.no_amp,
        )

    records = select_records(
        read_manifest(args.manifest),
        load_split_patients(args.splits, args.split),
        args.max_records,
        args.seed,
    )
    details = []
    for number, record in enumerate(records, start=1):
        print(
            f"[{number}/{len(records)}] {record.patient_id} "
            f"B{record.beam_idx} CP{record.cp_idx:03d}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = predict_record_volume(
            model,
            record,
            device=device,
            patch_size_zyx=patch_size,
            dose_scale=dose_scale,
            ct_clip=ct_clip,
            overlap=args.overlap,
            batch_size=args.batch_size,
            amp=not args.no_amp,
            skip_empty_aperture=args.skip_empty_aperture,
            mask_outside_body=args.mask_outside_body,
            include_density=include_density,
            include_physics_priors=include_physics_priors,
            include_radiological_depth=include_radiological_depth,
            pad_to_batch_size=args.pad_batch,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        target = np.asarray(load_mha_array(record.dose_path), dtype=np.float32)
        target_max = float(target.max())
        masked_mae_value = masked_beam_mae(prediction, target)
        masked_mae_result = (
            masked_mae_value if np.isfinite(masked_mae_value) else None
        )
        header = read_mha_header(record.ct_path)
        spacing_xyz = tuple(float(value) for value in header["ElementSpacing"].split())
        condition = lookup_condition(record)
        idd_value = idd_curve_distance(
            prediction,
            target,
            beam_direction(condition.gantry_angle_deg),
            spacing_xyz,
        )
        nrmse = (
            float(np.sqrt(np.mean((prediction - target) ** 2)) / target_max)
            if target_max > 0
            else None
        )
        details.append(
            {
                "patient_id": record.patient_id,
                "beam_idx": record.beam_idx,
                "cp_idx": record.cp_idx,
                "masked_beam_mae": masked_mae_result,
                "idd_curve_distance": idd_value if np.isfinite(idd_value) else None,
                "normalized_rmse": nrmse,
                "seconds": seconds,
                "target_max": target_max,
                "prediction_max": float(prediction.max()),
            }
        )

    patient_metrics = aggregate_by_patient(details)

    def valid_values(items: list[dict], key: str) -> list[float]:
        return [item[key] for item in items if item[key] is not None]

    record_mae = valid_values(details, "masked_beam_mae")
    record_nrmse = valid_values(details, "normalized_rmse")
    record_idd = valid_values(details, "idd_curve_distance")
    patient_mae = valid_values(patient_metrics, "masked_beam_mae")
    patient_nrmse = valid_values(patient_metrics, "normalized_rmse")
    patient_idd = valid_values(patient_metrics, "idd_curve_distance")
    confidence_intervals = {
        key: bootstrap_confidence_interval(
            patient_metrics,
            key,
            samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            seed=args.seed + index,
        )
        for index, key in enumerate(
            ("masked_beam_mae", "normalized_rmse", "idd_curve_distance")
        )
    }
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "device": str(device),
        "patch_size_zyx": list(patch_size),
        "records": len(details),
        "patients": len(patient_metrics),
        "include_density": include_density,
        "include_physics_priors": include_physics_priors,
        "include_radiological_depth": include_radiological_depth,
        "aggregation": "beam-to-patient, then patient-to-submission",
        "sampling": "patient-balanced, beam-balanced, control-point arc quantiles",
        "mean_masked_beam_mae": float(np.mean(patient_mae)) if patient_mae else None,
        "mean_normalized_rmse": float(np.mean(patient_nrmse)) if patient_nrmse else None,
        "mean_idd_curve_distance": float(np.mean(patient_idd)) if patient_idd else None,
        "record_mean_masked_beam_mae": float(np.mean(record_mae)) if record_mae else None,
        "record_mean_normalized_rmse": float(np.mean(record_nrmse)) if record_nrmse else None,
        "record_mean_idd_curve_distance": float(np.mean(record_idd)) if record_idd else None,
        "confidence_intervals": confidence_intervals,
        "mean_seconds": float(np.mean([item["seconds"] for item in details])),
        "skip_empty_aperture": args.skip_empty_aperture,
        "mask_outside_body": args.mask_outside_body,
        "torch_compile": args.torch_compile,
        "pad_batch": args.pad_batch,
        "patient_metrics": patient_metrics,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "details"}, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
