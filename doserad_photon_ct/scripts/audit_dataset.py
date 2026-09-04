#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from doserad_photon_ct import audit_dataset, build_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit DoseRAD2026 photon-CT files and build a training manifest."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT.parent / "data",
        help="Dataset root containing photon/training (default: ../data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
        help="Directory for audit.json and manifest.csv",
    )
    parser.add_argument(
        "--check-headers",
        action="store_true",
        help="Compare CT and dose MHA geometry without decompressing voxel data",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return a non-zero status unless the expected complete patient count is present",
    )
    parser.add_argument("--expected-patients", type=int, default=75)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = audit_dataset(args.data_root, check_headers=args.check_headers)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit_path = args.output_dir / "audit.json"
    manifest_path = args.output_dir / "manifest.csv"
    audit_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    row_count = build_manifest(summary, manifest_path)

    print(f"Patients found:       {summary.patient_count}")
    print(f"Complete photon-CT:   {summary.complete_patient_count}")
    print(f"Incomplete:           {summary.incomplete_patient_count}")
    print(f"Expected dose maps:   {summary.expected_dose_count}")
    print(f"Available dose maps:  {summary.existing_expected_dose_count}")
    print(f"Missing dose maps:    {summary.missing_dose_count}")
    print(f"Manifest rows:        {row_count}")
    print(f"Audit report:         {audit_path}")
    print(f"Training manifest:    {manifest_path}")

    if summary.incomplete_patient_count:
        print("\nIncomplete patients (normal while download is running):")
        for patient in summary.patients:
            if not patient.complete_for_photon_ct:
                print(
                    f"- {patient.patient_id}: CT={patient.ct_exists}, "
                    f"metadata={patient.metadata_exists}, "
                    f"dose={patient.existing_dose_count}/{patient.expected_dose_count}, "
                    f"errors={len(patient.errors)}"
                )
    if args.require_complete and (
        summary.complete_patient_count != args.expected_patients
        or summary.incomplete_patient_count > 0
        or summary.missing_dose_count > 0
    ):
        print(
            f"\nDataset is not complete: require {args.expected_patients} complete patients "
            "with no missing dose maps."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
