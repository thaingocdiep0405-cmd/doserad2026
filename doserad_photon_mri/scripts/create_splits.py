#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic patient-level train/validation splits."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "splits.json",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_patients(manifest_path: Path) -> dict[str, str]:
    patients: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            patient_id = row["patient_id"]
            group = row["anatomy_group"]
            previous = patients.setdefault(patient_id, group)
            if previous != group:
                raise ValueError(f"patient {patient_id} belongs to multiple anatomy groups")
    if not patients:
        raise ValueError("manifest contains no complete patients")
    return patients


def make_splits(
    patients: dict[str, str], validation_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")

    grouped: dict[str, list[str]] = defaultdict(list)
    for patient_id, group in patients.items():
        grouped[group].append(patient_id)

    rng = random.Random(seed)
    train: list[str] = []
    validation: list[str] = []
    for group in sorted(grouped):
        group_patients = sorted(grouped[group])
        rng.shuffle(group_patients)
        if len(group_patients) == 1:
            validation_count = 0
        else:
            validation_count = max(1, round(len(group_patients) * validation_fraction))
            validation_count = min(validation_count, len(group_patients) - 1)
        validation.extend(group_patients[:validation_count])
        train.extend(group_patients[validation_count:])
    return sorted(train), sorted(validation)


def main() -> int:
    args = parse_args()
    patients = load_patients(args.manifest)
    train, validation = make_splits(patients, args.validation_fraction, args.seed)
    payload = {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "note": "Patient-level split; regenerate after the full dataset download finishes.",
        "train": train,
        "validation": validation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Train patients:      {len(train)}")
    print(f"Validation patients: {len(validation)}")
    print(f"Split file:          {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
