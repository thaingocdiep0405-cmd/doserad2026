"""Training components shared by the DoseRAD2026 proton CT and MRI tasks."""

from .conditioning import ProtonCondition, build_proton_channels
from .data import ProtonPatchDataset, ProtonRecord, read_manifest
from .inference import predict_conditioned_arrays, predict_record_volume

__all__ = [
    "ProtonCondition",
    "ProtonPatchDataset",
    "ProtonRecord",
    "build_proton_channels",
    "predict_conditioned_arrays",
    "predict_record_volume",
    "read_manifest",
]
