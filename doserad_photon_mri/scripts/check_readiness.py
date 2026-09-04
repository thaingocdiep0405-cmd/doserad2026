#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from doserad_photon_mri import audit_dataset  # noqa: E402


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def main() -> int:
    data_root = PROJECT_ROOT.parent / "data"
    summary = audit_dataset(data_root)
    cache_root = data_root / ".cache" / "huggingface" / "download"
    incomplete_markers = []
    if cache_root.is_dir():
        incomplete_markers = list(cache_root.rglob("*.incomplete"))

    beam_parameters = data_root / "beam_parameters.json"
    baseline_repo = PROJECT_ROOT / "official" / "pyradplan-pb-baseline"
    submission_repo = PROJECT_ROOT / "official" / "example-submission"
    evaluation_repo = PROJECT_ROOT / "official" / "evaluation-setup"

    print("DoseRAD2026 photon-CT readiness")
    print(f"Architecture:                 {platform.machine()}")
    print(f"Patients discovered:         {summary.patient_count}/75")
    print(f"Complete photon-CT patients: {summary.complete_patient_count}/75")
    print(f"Missing expected dose maps:  {summary.missing_dose_count}")
    print(f"Incomplete transfer files:   {len(incomplete_markers)}")
    print(f"beam_parameters.json:        {yes_no(beam_parameters.is_file())}")
    print(f"SimpleITK installed:         {yes_no(importlib.util.find_spec('SimpleITK') is not None)}")
    print(f"pyRadPlan installed:         {yes_no(importlib.util.find_spec('pyRadPlan') is not None)}")
    print(f"Official baseline cloned:    {yes_no(baseline_repo.is_dir())}")
    print(f"Submission template cloned:  {yes_no(submission_repo.is_dir())}")
    print(f"Evaluation code cloned:      {yes_no(evaluation_repo.is_dir())}")

    ready_for_full_training = (
        summary.complete_patient_count == 75
        and summary.incomplete_patient_count == 0
        and summary.missing_dose_count == 0
    )
    print(f"Ready for full training:     {yes_no(ready_for_full_training)}")
    if not ready_for_full_training:
        print("\nCurrent state is suitable for pipeline development, not the final training run.")
    if not beam_parameters.is_file():
        print(
            "The official pyRadPlan baseline expects data/beam_parameters.json; "
            "it is not present yet. Do not silently invent this calibration file."
        )
    if platform.machine() == "aarch64":
        print(
            "The official submission Dockerfile targets linux/amd64. Train locally on "
            "aarch64, but build/test the final submission as linux/amd64."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
