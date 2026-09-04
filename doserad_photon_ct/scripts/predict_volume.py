#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from doserad_photon_ct.dataset import read_manifest  # noqa: E402
from doserad_photon_ct.inference import predict_record_volume  # noqa: E402
from doserad_photon_ct.mha import load_mha_array, write_mha_array_like  # noqa: E402
from doserad_photon_ct.model import ModelConfig, PhotonDoseUNet3D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one full photon-CT dose volume.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts/manifest.csv")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--patient-id")
    parser.add_argument("--beam-idx", type=int)
    parser.add_argument("--cp-idx", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--patch-size", type=int, nargs=3, metavar=("Z", "Y", "X"))
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compress", action="store_true")
    return parser.parse_args()


def choose_record(args: argparse.Namespace):
    records = read_manifest(args.manifest)
    if args.patient_id is not None:
        matches = [
            record
            for record in records
            if record.patient_id == args.patient_id
            and (args.beam_idx is None or record.beam_idx == args.beam_idx)
            and (args.cp_idx is None or record.cp_idx == args.cp_idx)
        ]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one matching record, found {len(matches)}")
        return matches[0]
    if not 0 <= args.row < len(records):
        raise IndexError(f"row must be between 0 and {len(records) - 1}")
    return records[args.row]


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    training_config = checkpoint["training_config"]
    model = PhotonDoseUNet3D(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device).eval()

    record = choose_record(args)
    patch_size = tuple(args.patch_size or training_config["patch_size"])
    dose_scale = float(training_config["dose_scale"])
    ct_clip = tuple(float(value) for value in training_config["ct_clip"])
    print(
        f"Predicting {record.patient_id} B{record.beam_idx} CP{record.cp_idx:03d} on {device}",
        flush=True,
    )
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
        include_density=int(checkpoint["model_config"].get("in_channels", 6)) >= 7,
        include_physics_priors=bool(
            checkpoint["model_config"].get("physics_priors", False)
        ),
        include_radiological_depth=bool(
            checkpoint["model_config"].get("radiological_depth", False)
        ),
    )
    write_mha_array_like(args.output, prediction, record.ct_path, compress=args.compress)

    result = {
        "patient_id": record.patient_id,
        "beam_idx": record.beam_idx,
        "cp_idx": record.cp_idx,
        "output": str(args.output.resolve()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
    }
    if record.dose_path.is_file():
        target = np.asarray(load_mha_array(record.dose_path), dtype=np.float32)
        target_max = float(target.max())
        mask = target >= 0.1 * target_max
        result["masked_beam_mae"] = (
            float(np.mean(np.abs(prediction[mask] - target[mask])) / target_max)
            if target_max > 0 and mask.any()
            else None
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
