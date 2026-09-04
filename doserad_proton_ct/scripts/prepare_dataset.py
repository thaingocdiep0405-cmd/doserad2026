#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
sys.path.insert(0, str(WORKSPACE / "doserad_photon_ct" / "src"))

from doserad_photon_ct.mha import load_mha_array  # noqa: E402


FIELDS = (
    "patient_id", "anatomy_group", "ct_path", "mr_path", "metadata_path", "dose_path",
    "beam_idx", "ray_idx", "beamlet_idx", "gantry_angle_deg",
    "source_x", "source_y", "source_z", "target_x", "target_y", "target_z",
    "energy_mev", "sigma_energy_mev", "sigma_spot_mm",
)


def anatomy_group(patient_id: str) -> str:
    match = re.match(r"^\d([A-Za-z]+)\d+$", patient_id)
    return match.group(1).upper() if match else "UNKNOWN"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=WORKSPACE / "data" / "proton" / "training")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "manifest.csv")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "artifacts" / "splits.json")
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "artifacts" / "audit.json")
    parser.add_argument("--reference-splits", type=Path, default=WORKSPACE / "doserad_photon_ct" / "artifacts" / "splits.json")
    args = parser.parse_args()

    beam_parameters = json.loads((args.data_root / "beam_parameters.json").read_text())
    energy_table = {
        round(float(item["energy_mev"]), 4): item
        for item in beam_parameters["proton"]["energy_table"]
    }
    patients = sorted(path for path in args.data_root.iterdir() if path.is_dir() and path.name.startswith("1"))
    errors: list[str] = []
    rows: list[dict[str, object]] = []
    patient_row_ranges: dict[str, tuple[int, int]] = {}
    for patient_dir in patients:
        patient_id = patient_dir.name
        ct_path = patient_dir / "image" / "ct.mha"
        mr_path = patient_dir / "image" / "mr.mha"
        metadata_path = patient_dir / f"{patient_id}.json"
        if not (ct_path.is_file() and mr_path.is_file() and metadata_path.is_file()):
            errors.append(f"{patient_id}: CT, MRI or metadata missing")
            continue
        plan = json.loads(metadata_path.read_text())
        start = len(rows)
        expected_names: set[str] = set()
        for beam in plan["beams"]:
            for ray in beam["rays"]:
                for beamlet in ray["beamlets"]:
                    name = f'Dose_B{beam["beam_idx"]}_R{ray["ray_idx"]}_L{beamlet["beamlet_idx"]}.mha'
                    expected_names.add(name)
                    dose_path = patient_dir / "dose" / name
                    energy = float(beamlet["energy"])
                    parameters = energy_table.get(round(energy, 4))
                    if parameters is None:
                        errors.append(f"{patient_id}: energy {energy} absent from table")
                        continue
                    if not dose_path.is_file():
                        errors.append(f"{patient_id}: missing {name}")
                        continue
                    row: dict[str, object] = {
                        "patient_id": patient_id,
                        "anatomy_group": anatomy_group(patient_id),
                        "ct_path": str(ct_path.resolve()),
                        "mr_path": str(mr_path.resolve()),
                        "metadata_path": str(metadata_path.resolve()),
                        "dose_path": str(dose_path.resolve()),
                        "beam_idx": int(beam["beam_idx"]),
                        "ray_idx": int(ray["ray_idx"]),
                        "beamlet_idx": int(beamlet["beamlet_idx"]),
                        "gantry_angle_deg": float(beam["gantry_angle"]),
                        "energy_mev": energy,
                        "sigma_energy_mev": float(parameters["sigma_energy_mev"]),
                        "sigma_spot_mm": float(parameters["sigma_spot_mm"]),
                    }
                    for prefix, values in (("source", ray["ray_source"]), ("target", ray["ray_target"])):
                        for axis, value in zip("xyz", values):
                            row[f"{prefix}_{axis}"] = float(value)
                    rows.append(row)
        actual_names = {path.name for path in (patient_dir / "dose").glob("Dose_B*_R*_L*.mha")}
        if actual_names != expected_names:
            errors.append(f"{patient_id}: expected {len(expected_names)}, found {len(actual_names)} dose files")
        patient_row_ranges[patient_id] = (start, len(rows))

    if errors:
        raise RuntimeError("dataset is incomplete:\n" + "\n".join(errors[:50]))
    if len(patients) != 75 or len(rows) != 81_000:
        raise RuntimeError(f"expected 75 patients/81000 rows, got {len(patients)}/{len(rows)}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    reference = json.loads(args.reference_splits.read_text())
    all_ids = {path.name for path in patients}
    split_ids = set(reference["train"]) | set(reference["validation"])
    if split_ids != all_ids or set(reference["train"]) & set(reference["validation"]):
        raise RuntimeError("reference Photon split does not exactly match Proton patients")
    split_payload = {
        "seed": reference.get("seed", 2026),
        "note": "Same patient-level split as Photon; validation patients are never used for gradients.",
        "train": reference["train"],
        "validation": reference["validation"],
    }
    args.splits.write_text(json.dumps(split_payload, indent=2) + "\n")

    maxima = []
    for patient_id in sorted(all_ids):
        begin, end = patient_row_ranges[patient_id]
        sample = rows[begin + (end - begin) // 2]
        maxima.append(float(np.asarray(load_mha_array(Path(str(sample["dose_path"]))), dtype=np.float32).max()))
    positive = [value for value in maxima if value > 0]
    audit = {
        "patient_count": len(patients),
        "record_count": len(rows),
        "ct_count": len(patients),
        "mr_count": len(patients),
        "train_patients": len(reference["train"]),
        "validation_patients": len(reference["validation"]),
        "sample_max_min": min(positive),
        "sample_max_median": float(np.median(positive)),
        "sample_max_max": max(positive),
    }
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
