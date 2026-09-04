from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from doserad_photon_ct.dataset import ManifestRecord  # noqa: E402
from doserad_photon_ct.metrics import (  # noqa: E402
    beam_direction,
    idd_curve_distance,
    masked_beam_mae,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_module("photon_ct_evaluation", PROJECT_ROOT / "scripts/evaluate_checkpoint.py")


def fake_record(patient_id: str, beam_idx: int, cp_idx: int) -> ManifestRecord:
    root = Path("/tmp") / patient_id
    return ManifestRecord(
        patient_id=patient_id,
        anatomy_group="ABB",
        ct_path=root / "ct.mha",
        metadata_path=root / "plan.json",
        dose_path=root / f"Dose_B{beam_idx}_CP{cp_idx:03d}.mha",
        beam_idx=beam_idx,
        cp_idx=cp_idx,
    )


def test_stratified_selection_balances_patients_beams_and_arc() -> None:
    records = [
        fake_record(patient, beam, cp)
        for patient in ("1ABB001", "1ABB002")
        for beam in range(3)
        for cp in range(10)
    ]
    selected = evaluation.select_records(
        records, {"1ABB001", "1ABB002"}, maximum=12, seed=2026
    )
    assert len(selected) == 12
    for patient in ("1ABB001", "1ABB002"):
        patient_records = [item for item in selected if item.patient_id == patient]
        assert len(patient_records) == 6
        assert {beam: sum(item.beam_idx == beam for item in patient_records) for beam in range(3)} == {
            0: 2,
            1: 2,
            2: 2,
        }
        for beam in range(3):
            cps = [item.cp_idx for item in patient_records if item.beam_idx == beam]
            assert max(cps) - min(cps) >= 4


def test_bootstrap_interval_and_patient_aggregation() -> None:
    details = [
        {"patient_id": "A", "masked_beam_mae": 1.0, "normalized_rmse": 2.0, "idd_curve_distance": 3.0},
        {"patient_id": "A", "masked_beam_mae": 3.0, "normalized_rmse": 4.0, "idd_curve_distance": 5.0},
        {"patient_id": "B", "masked_beam_mae": 10.0, "normalized_rmse": 20.0, "idd_curve_distance": 30.0},
    ]
    patients = evaluation.aggregate_by_patient(details)
    assert patients[0]["masked_beam_mae"] == 2.0
    interval = evaluation.bootstrap_confidence_interval(
        patients,
        "masked_beam_mae",
        samples=1000,
        confidence_level=0.95,
        seed=2026,
    )
    assert interval is not None
    assert interval["mean"] == 6.0
    assert interval["lower"] <= 6.0 <= interval["upper"]


def test_local_beam_metrics_match_official_implementation() -> None:
    pytest.importorskip("SimpleITK")
    official = load_module(
        "official_metrics_beam",
        PROJECT_ROOT / "official/evaluation-setup/doserad2026_evaluator/metrics_beam.py",
    )
    rng = np.random.default_rng(2026)
    target = rng.random((7, 19, 23), dtype=np.float32)
    prediction = np.clip(target + 0.03 * rng.standard_normal(target.shape), 0.0, None)
    direction = beam_direction(37.0)
    spacing = np.asarray((2.0, 2.0, 2.0), dtype=np.float64)

    np.testing.assert_allclose(
        masked_beam_mae(prediction, target),
        official.masked_beam_mae(prediction, target),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        idd_curve_distance(prediction, target, direction, tuple(spacing)),
        official.idd_curve_distance(prediction, target, direction, spacing),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
