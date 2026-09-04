from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


DOSE_NAME_RE = re.compile(r"^Dose_B(?P<beam>\d+)_CP(?P<cp>\d+)\.mha$")


@dataclass
class PatientAudit:
    patient_id: str
    anatomy_group: str
    patient_dir: str
    metadata_path: str
    ct_path: str
    mr_path: str
    metadata_exists: bool
    ct_exists: bool
    mr_exists: bool
    beam_count: int = 0
    expected_dose_count: int = 0
    existing_dose_count: int = 0
    missing_dose_count: int = 0
    unexpected_dose_count: int = 0
    complete_for_photon_mri: bool = False
    errors: list[str] = field(default_factory=list)
    missing_dose_files: list[str] = field(default_factory=list)
    unexpected_dose_files: list[str] = field(default_factory=list)


@dataclass
class AuditSummary:
    data_root: str
    patient_count: int
    complete_patient_count: int
    incomplete_patient_count: int
    expected_dose_count: int
    existing_expected_dose_count: int
    missing_dose_count: int
    unexpected_dose_count: int
    patients: list[PatientAudit]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["patients"] = [asdict(patient) for patient in self.patients]
        return result


def anatomy_group(patient_id: str) -> str:
    """Return a stable grouping key such as ABB, HNB or THB."""
    match = re.match(r"^\d([A-Za-z]+)\d+$", patient_id)
    return match.group(1).upper() if match else "UNKNOWN"


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("beams"), list):
        raise ValueError("metadata must be an object containing a 'beams' list")
    return payload


def iter_control_points(metadata: dict[str, Any]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for beam_position, beam in enumerate(metadata["beams"]):
        if not isinstance(beam, dict):
            raise ValueError(f"beam at position {beam_position} is not an object")
        beam_idx = beam.get("beam_idx")
        control_points = beam.get("control_points")
        if not isinstance(beam_idx, int):
            raise ValueError(f"beam at position {beam_position} has invalid beam_idx")
        if not isinstance(control_points, list):
            raise ValueError(f"beam {beam_idx} has no control_points list")

        leaf_pairs = beam.get("num_mlc_leaf_pairs")
        for cp_position, control_point in enumerate(control_points):
            if not isinstance(control_point, dict):
                raise ValueError(f"beam {beam_idx}, CP position {cp_position} is not an object")
            cp_idx = control_point.get("cp_idx")
            if not isinstance(cp_idx, int):
                raise ValueError(f"beam {beam_idx}, CP position {cp_position} has invalid cp_idx")
            if isinstance(leaf_pairs, int):
                left = control_point.get("mlc_left_int_mm")
                right = control_point.get("mlc_right_int_mm")
                if not isinstance(left, list) or len(left) != leaf_pairs:
                    raise ValueError(f"beam {beam_idx}, CP {cp_idx} has invalid left MLC length")
                if not isinstance(right, list) or len(right) != leaf_pairs:
                    raise ValueError(f"beam {beam_idx}, CP {cp_idx} has invalid right MLC length")
            yield beam, control_point


def expected_dose_name(beam_idx: int, cp_idx: int) -> str:
    return f"Dose_B{beam_idx}_CP{cp_idx:03d}.mha"


def read_mha_header(path: Path, max_bytes: int = 65_536) -> dict[str, str]:
    """Read a local MetaImage header without decompressing the voxel payload."""
    header: dict[str, str] = {}
    with path.open("rb") as handle:
        data = handle.read(max_bytes)
    marker = b"ElementDataFile"
    marker_position = data.find(marker)
    if marker_position < 0:
        raise ValueError("ElementDataFile is missing from MHA header")
    line_end = data.find(b"\n", marker_position)
    if line_end < 0:
        line_end = len(data)
    text = data[:line_end].decode("ascii", errors="strict")
    for raw_line in text.splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        header[key.strip()] = value.strip()
    for required in ("NDims", "DimSize", "ElementSpacing", "ElementType", "ElementDataFile"):
        if required not in header:
            raise ValueError(f"{required} is missing from MHA header")
    return header


def audit_patient(patient_dir: Path, check_headers: bool = False) -> PatientAudit:
    patient_id = patient_dir.name
    metadata_path = patient_dir / f"{patient_id}.json"
    ct_path = patient_dir / "image" / "ct.mha"
    mr_path = patient_dir / "image" / "mr.mha"
    dose_dir = patient_dir / "dose"

    result = PatientAudit(
        patient_id=patient_id,
        anatomy_group=anatomy_group(patient_id),
        patient_dir=str(patient_dir.resolve()),
        metadata_path=str(metadata_path.resolve()),
        ct_path=str(ct_path.resolve()),
        mr_path=str(mr_path.resolve()),
        metadata_exists=metadata_path.is_file(),
        ct_exists=ct_path.is_file(),
        mr_exists=mr_path.is_file(),
    )

    metadata: dict[str, Any] | None = None
    expected_names: set[str] = set()
    if result.metadata_exists:
        try:
            metadata = load_metadata(metadata_path)
            result.beam_count = len(metadata["beams"])
            for beam, cp in iter_control_points(metadata):
                expected_names.add(expected_dose_name(beam["beam_idx"], cp["cp_idx"]))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            result.errors.append(f"metadata: {exc}")
    else:
        result.errors.append("metadata file is missing")

    if not result.mr_exists:
        result.errors.append("MRI image is missing")

    existing_names = {
        path.name
        for path in dose_dir.glob("Dose_B*_CP*.mha")
        if path.is_file() and DOSE_NAME_RE.match(path.name)
    }
    missing_names = sorted(expected_names - existing_names)
    unexpected_names = sorted(existing_names - expected_names) if metadata is not None else sorted(existing_names)

    result.expected_dose_count = len(expected_names)
    result.existing_dose_count = len(existing_names & expected_names)
    result.missing_dose_count = len(missing_names)
    result.unexpected_dose_count = len(unexpected_names)
    result.missing_dose_files = missing_names
    result.unexpected_dose_files = unexpected_names

    if check_headers and result.mr_exists:
        try:
            mr_header = read_mha_header(mr_path)
            geometry_keys = ("NDims", "DimSize", "ElementSpacing", "Offset", "TransformMatrix")
            for dose_name in sorted(existing_names & expected_names):
                dose_path = dose_dir / dose_name
                dose_header = read_mha_header(dose_path)
                mismatches = [
                    key for key in geometry_keys if mr_header.get(key) != dose_header.get(key)
                ]
                if mismatches:
                    result.errors.append(
                        f"{dose_name}: geometry differs from MRI ({', '.join(mismatches)})"
                    )
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            result.errors.append(f"MHA header: {exc}")

    result.complete_for_photon_mri = (
        result.metadata_exists
        and result.mr_exists
        and result.expected_dose_count > 0
        and result.missing_dose_count == 0
        and result.unexpected_dose_count == 0
        and not result.errors
    )
    return result


def audit_dataset(data_root: Path, check_headers: bool = False) -> AuditSummary:
    training_root = data_root.resolve() / "photon" / "training"
    if not training_root.is_dir():
        raise FileNotFoundError(f"photon training directory not found: {training_root}")

    patients = [
        audit_patient(path, check_headers=check_headers)
        for path in sorted(training_root.iterdir())
        if path.is_dir() and not path.name.startswith(".")
    ]
    complete = sum(patient.complete_for_photon_mri for patient in patients)
    return AuditSummary(
        data_root=str(data_root.resolve()),
        patient_count=len(patients),
        complete_patient_count=complete,
        incomplete_patient_count=len(patients) - complete,
        expected_dose_count=sum(patient.expected_dose_count for patient in patients),
        existing_expected_dose_count=sum(patient.existing_dose_count for patient in patients),
        missing_dose_count=sum(patient.missing_dose_count for patient in patients),
        unexpected_dose_count=sum(patient.unexpected_dose_count for patient in patients),
        patients=patients,
    )


MANIFEST_FIELDS = (
    "patient_id",
    "anatomy_group",
    "image_path",
    "metadata_path",
    "dose_path",
    "beam_idx",
    "cp_idx",
    "gantry_angle",
    "sad",
    "iso_x",
    "iso_y",
    "iso_z",
    "num_mlc_leaf_pairs",
)


def build_manifest(summary: AuditSummary, output_path: Path) -> int:
    """Write one row per beam/control-point for complete photon-MRI patients."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for patient in summary.patients:
            if not patient.complete_for_photon_mri:
                continue
            metadata = load_metadata(Path(patient.metadata_path))
            dose_dir = Path(patient.patient_dir) / "dose"
            for beam, cp in iter_control_points(metadata):
                iso_center = beam.get("iso_center", [None, None, None])
                writer.writerow(
                    {
                        "patient_id": patient.patient_id,
                        "anatomy_group": patient.anatomy_group,
                        "image_path": patient.mr_path,
                        "metadata_path": patient.metadata_path,
                        "dose_path": str(
                            (dose_dir / expected_dose_name(beam["beam_idx"], cp["cp_idx"])).resolve()
                        ),
                        "beam_idx": beam["beam_idx"],
                        "cp_idx": cp["cp_idx"],
                        "gantry_angle": cp.get("gantry_angle"),
                        "sad": beam.get("SAD"),
                        "iso_x": iso_center[0],
                        "iso_y": iso_center[1],
                        "iso_z": iso_center[2],
                        "num_mlc_leaf_pairs": beam.get("num_mlc_leaf_pairs"),
                    }
                )
                row_count += 1
    return row_count
