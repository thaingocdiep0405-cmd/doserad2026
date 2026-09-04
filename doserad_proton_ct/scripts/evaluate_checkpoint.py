#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE / "doserad_photon_ct" / "src"))

from doserad_photon_ct.dataset_index import read_mha_header  # noqa: E402
from doserad_photon_ct.metrics import idd_curve_distance  # noqa: E402
from doserad_photon_ct.mha import load_mha_array  # noqa: E402
from doserad_photon_ct.model import ModelConfig, PhotonDoseUNet3D  # noqa: E402
from doserad_proton.data import ProtonRecord, load_split_patients, read_manifest  # noqa: E402
from doserad_proton.inference import predict_record_volume  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-volume Proton checkpoint evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--modality", choices=("ct", "mri"), required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "manifest.csv")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "artifacts" / "splits.json")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-patients", type=int, default=15)
    parser.add_argument("--records-per-patient", type=int, default=3)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dose-scale", type=float)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--ray-gate-threshold", type=float, default=0.0)
    parser.add_argument("--relative-cutoff", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select_records(args: argparse.Namespace) -> list[ProtonRecord]:
    patients = load_split_patients(args.splits, args.split)
    grouped: dict[str, list[ProtonRecord]] = defaultdict(list)
    for record in read_manifest(args.manifest):
        if record.patient_id in patients:
            grouped[record.patient_id].append(record)
    patient_ids = sorted(grouped)
    random.Random(args.seed + 17).shuffle(patient_ids)
    selected: list[ProtonRecord] = []
    for patient_id in patient_ids[: args.max_patients]:
        records = sorted(
            grouped[patient_id],
            key=lambda item: (item.beam_idx, item.ray_idx, item.beamlet_idx),
        )
        count = min(args.records_per_patient, len(records))
        indices = [
            min(len(records) - 1, int((slot + 0.5) * len(records) / count))
            for slot in range(count)
        ]
        selected.extend(records[index] for index in indices)
    return selected


def direction_of(record: ProtonRecord) -> np.ndarray:
    source = np.asarray(record.condition.ray_source_xyz, dtype=np.float64)
    target = np.asarray(record.condition.ray_target_xyz, dtype=np.float64)
    direction = target - source
    return direction / np.linalg.norm(direction)


def aggregate(per_record: list[dict]) -> dict:
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for row in per_record:
        by_patient[row["patient_id"]].append(row)
    per_patient = []
    metric_names = ("masked_mae", "nrmse", "idd_distance", "scale_ratio", "seconds")
    for patient_id, rows in sorted(by_patient.items()):
        item = {"patient_id": patient_id, "records": len(rows)}
        for name in metric_names:
            item[name] = float(np.nanmean([row[name] for row in rows]))
        per_patient.append(item)
    summary = {"patients": len(per_patient), "records": len(per_record)}
    for name in metric_names:
        values = np.asarray([row[name] for row in per_patient], dtype=np.float64)
        summary[name] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "median": float(np.nanmedian(values)),
        }
    return {"summary": summary, "per_patient": per_patient}


def main() -> int:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = PhotonDoseUNet3D(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    training_config = checkpoint.get("training_config", {})
    dose_scale = args.dose_scale or float(training_config["dose_scale"])
    records = select_records(args)
    if not records:
        raise RuntimeError("no records selected")

    rows = []
    for index, record in enumerate(records, 1):
        print(
            f"[{index}/{len(records)}] {record.patient_id} "
            f"B{record.beam_idx} R{record.ray_idx} L{record.beamlet_idx}",
            flush=True,
        )
        prediction, seconds = predict_record_volume(
            model,
            record,
            modality=args.modality,
            device=device,
            patch_size_zyx=tuple(args.patch_size),
            dose_scale=dose_scale,
            overlap=args.overlap,
            batch_size=args.batch_size,
            amp=not args.no_amp,
            ray_gate_threshold=args.ray_gate_threshold,
            relative_cutoff=args.relative_cutoff,
        )
        target = np.asarray(load_mha_array(record.dose_path), dtype=np.float32)
        maximum = float(target.max())
        mask = target >= 0.1 * maximum
        masked_mae = float(np.mean(np.abs(prediction[mask] - target[mask])) / maximum)
        nrmse = float(np.sqrt(np.mean((prediction - target) ** 2)) / maximum)
        image_path = record.ct_path if args.modality == "ct" else record.mr_path
        spacing = np.asarray(
            [float(value) for value in read_mha_header(image_path)["ElementSpacing"].split()],
            dtype=np.float64,
        )
        idd = idd_curve_distance(prediction, target, direction_of(record), spacing)
        target_mean = float(target[mask].mean())
        scale_ratio = float(prediction[mask].mean() / max(target_mean, 1.0e-12))
        row = {
            "patient_id": record.patient_id,
            "beam_idx": record.beam_idx,
            "ray_idx": record.ray_idx,
            "beamlet_idx": record.beamlet_idx,
            "energy_mev": record.condition.energy_mev,
            "masked_mae": masked_mae,
            "nrmse": nrmse,
            "idd_distance": idd,
            "scale_ratio": scale_ratio,
            "seconds": seconds,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "modality": args.modality,
        "split": args.split,
        "dose_scale": dose_scale,
        "patch_size": list(args.patch_size),
        "overlap": args.overlap,
        "batch_size": args.batch_size,
        "ray_gate_threshold": args.ray_gate_threshold,
        "relative_cutoff": args.relative_cutoff,
        **aggregate(rows),
        "per_record": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
