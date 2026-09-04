#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from doserad_photon_mri import load_mha_array  # noqa: E402
from doserad_photon_mri.dataset import load_split_patients  # noqa: E402


PERCENTILES = (0.1, 1.0, 50.0, 99.0, 99.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute lightweight CT/dose statistics from manifest samples."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sample_stats.json",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "splits.json",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Only use patients from this split; set to 'all' to disable filtering",
    )
    parser.add_argument("--max-patients", type=int, default=3)
    parser.add_argument("--doses-per-patient", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def array_stats(array: np.ndarray, include_body: bool = False) -> dict[str, object]:
    result: dict[str, object] = {
        "shape_zyx": list(array.shape),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "nonzero_fraction": float(np.count_nonzero(array) / array.size),
        "percentiles": {
            str(percentile): float(value)
            for percentile, value in zip(PERCENTILES, np.percentile(array, PERCENTILES))
        },
    }
    if include_body:
        body = array[array > -1000.0]
        result["body_voxel_fraction"] = float(body.size / array.size)
        result["body_percentiles"] = {
            str(percentile): float(value)
            for percentile, value in zip(PERCENTILES, np.percentile(body, PERCENTILES))
        }
    return result


def main() -> int:
    args = parse_args()
    allowed_patients = (
        None if args.split == "all" else load_split_patients(args.splits, args.split)
    )
    rows_by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if allowed_patients is not None and row["patient_id"] not in allowed_patients:
                continue
            rows_by_patient[row["patient_id"]].append(row)
    if not rows_by_patient:
        raise ValueError("manifest has no complete patients")

    rng = random.Random(args.seed)
    patient_ids = sorted(rows_by_patient)
    rng.shuffle(patient_ids)
    patient_ids = patient_ids[: args.max_patients]

    results: list[dict[str, object]] = []
    for patient_id in patient_ids:
        rows = rows_by_patient[patient_id]
        selected_rows = rng.sample(rows, min(args.doses_per_patient, len(rows)))
        print(f"Reading CT for {patient_id}...")
        ct_array = load_mha_array(Path(rows[0]["ct_path"]))
        dose_results = []
        for row in selected_rows:
            print(f"Reading {Path(row['dose_path']).name}...")
            dose_array = load_mha_array(Path(row["dose_path"]))
            dose_results.append(
                {
                    "beam_idx": int(row["beam_idx"]),
                    "cp_idx": int(row["cp_idx"]),
                    "path": row["dose_path"],
                    "stats": array_stats(dose_array),
                }
            )
        results.append(
            {
                "patient_id": patient_id,
                "ct": array_stats(ct_array, include_body=True),
                "doses": dose_results,
            }
        )

    payload = {
        "seed": args.seed,
        "split": args.split,
        "patient_count": len(results),
        "doses_per_patient": args.doses_per_patient,
        "warning": "Sampled dose statistics; CT statistics cover every selected patient.",
        "patients": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Statistics written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
