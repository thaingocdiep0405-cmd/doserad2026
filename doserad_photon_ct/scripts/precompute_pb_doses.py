"""Precompute pyRadPlan SVDPB doses for PB distillation.

Selects an evenly spaced subset of control points per beam for the chosen
split and writes one compressed ``<pid>_B<b>_CP<ccc>.npz`` (fp16 ``dose``
array, ZYX order matching the MHA layout) per control point. Work is
parallelized across patients; each worker pays the CT-load/body-seg setup
once per patient. Existing outputs are skipped, so the script is resumable.

Measured cost: ~19 s per control point per core (GB10).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "official" / "pyradplan-pb-baseline"))

from doserad_photon_ct.dataset import load_split_patients  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT.parent / "data"
    )
    parser.add_argument(
        "--splits", type=Path, default=PROJECT_ROOT / "artifacts/splits.json"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/pb_doses"
    )
    parser.add_argument(
        "--per-beam",
        type=int,
        default=10,
        help="Evenly spaced control points per beam",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--patients", nargs="*", help="Optional explicit patient subset"
    )
    return parser.parse_args()


def select_positions(count: int, wanted: int) -> list[int]:
    if wanted >= count:
        return list(range(count))
    return sorted({int(round(p)) for p in np.linspace(0, count - 1, wanted)})


def process_patient(job: tuple[str, str, str, int]) -> dict:
    patient_id, data_root_text, output_dir_text, per_beam = job
    import logging

    logging.basicConfig(level=logging.ERROR)
    for name in ("pyRadPlan", "baseline_photons"):
        logging.getLogger(name).setLevel(logging.ERROR)

    import SimpleITK as sitk
    from baseline_photons import _compute_cp_dose
    from pyRadPlan.ct import ct_from_file
    from pyRadPlan.cst import StructureSet

    data_root = Path(data_root_text)
    output_dir = Path(output_dir_text)
    patient_dir = data_root / "photon" / "training" / patient_id

    beam_params = json.loads((data_root / "beam_parameters.json").read_text())
    hlut = np.array(
        [tuple(e.values()) for e in beam_params["hu_to_density"]["entries"]],
        dtype=float,
    )
    plan = json.loads((patient_dir / f"{patient_id}.json").read_text())

    jobs = []
    for beam_json in plan["beams"]:
        control_points = beam_json["control_points"]
        for position in select_positions(len(control_points), per_beam):
            cp_json = control_points[position]
            name = (
                f"{patient_id}_B{int(beam_json['beam_idx'])}"
                f"_CP{int(cp_json['cp_idx']):03d}.npz"
            )
            if not (output_dir / name).exists():
                jobs.append((beam_json, cp_json, name))
    if not jobs:
        return {"patient": patient_id, "done": 0, "skipped": True, "errors": []}

    ct = ct_from_file(patient_dir / "image" / "ct.mha")
    cst = StructureSet(vois=[], ct_image=ct)
    cst.create_body_seg(voi_type="TARGET")

    done, errors = 0, []
    for beam_json, cp_json, name in jobs:
        try:
            pb = _compute_cp_dose(ct, cst, beam_json, cp_json, hlut=hlut)
            dose = sitk.GetArrayFromImage(pb).astype(np.float16)
            tmp = output_dir / (name + ".tmp.npz")
            np.savez_compressed(tmp, dose=dose)
            tmp.replace(output_dir / name)
            done += 1
        except Exception:
            errors.append(f"{name}: {traceback.format_exc(limit=1)}")
    return {"patient": patient_id, "done": done, "skipped": False, "errors": errors}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patients = sorted(
        args.patients
        if args.patients
        else load_split_patients(args.splits, args.split)
    )
    print(
        f"{len(patients)} patients, {args.per_beam}/beam, "
        f"{args.workers} workers -> {args.output_dir}",
        flush=True,
    )
    started = time.perf_counter()
    jobs = [
        (patient, str(args.data_root), str(args.output_dir), args.per_beam)
        for patient in patients
    ]
    total_done, total_errors = 0, []
    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(process_patient, jobs):
            total_done += result["done"]
            total_errors.extend(result["errors"])
            elapsed = time.perf_counter() - started
            print(
                f"[{elapsed/60:6.1f} min] {result['patient']}: "
                f"+{result['done']} maps"
                + (" (already complete)" if result["skipped"] else "")
                + (f", {len(result['errors'])} errors" if result["errors"] else ""),
                flush=True,
            )
    print(f"finished: {total_done} maps, {len(total_errors)} errors")
    for line in total_errors[:10]:
        print(line)
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
