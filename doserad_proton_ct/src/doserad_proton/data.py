from __future__ import annotations

import csv
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from doserad_photon_ct.conditioning import SpatialGeometry
from doserad_photon_ct.dataset import crop_with_padding
from doserad_photon_ct.dataset_index import read_mha_header
from doserad_photon_ct.mha import load_mha_array

from .conditioning import ProtonCondition, build_proton_channels, compute_proton_wepl


@dataclass(frozen=True)
class ProtonRecord:
    patient_id: str
    anatomy_group: str
    ct_path: Path
    mr_path: Path
    metadata_path: Path
    dose_path: Path
    beam_idx: int
    ray_idx: int
    beamlet_idx: int
    condition: ProtonCondition


def read_manifest(path: Path) -> list[ProtonRecord]:
    records: list[ProtonRecord] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            condition = ProtonCondition(
                gantry_angle_deg=float(row["gantry_angle_deg"]),
                ray_source_xyz=tuple(float(row[f"source_{axis}"]) for axis in "xyz"),
                ray_target_xyz=tuple(float(row[f"target_{axis}"]) for axis in "xyz"),
                energy_mev=float(row["energy_mev"]),
                sigma_energy_mev=float(row["sigma_energy_mev"]),
                sigma_spot_mm=float(row["sigma_spot_mm"]),
            )
            records.append(
                ProtonRecord(
                    patient_id=row["patient_id"],
                    anatomy_group=row["anatomy_group"],
                    ct_path=Path(row["ct_path"]),
                    mr_path=Path(row["mr_path"]),
                    metadata_path=Path(row["metadata_path"]),
                    dose_path=Path(row["dose_path"]),
                    beam_idx=int(row["beam_idx"]),
                    ray_idx=int(row["ray_idx"]),
                    beamlet_idx=int(row["beamlet_idx"]),
                    condition=condition,
                )
            )
    return records


def load_split_patients(path: Path, split: str) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get(split)
    if not isinstance(values, list):
        raise ValueError(f"split file has no list named {split!r}")
    return {str(value) for value in values}


class _VolumeCache:
    def __init__(self, max_items: int):
        self.max_items = max(0, int(max_items))
        self._items: OrderedDict[str, NDArray[np.generic]] = OrderedDict()

    def get(self, path: Path) -> NDArray[np.generic]:
        key = str(path.resolve())
        if key in self._items:
            value = self._items.pop(key)
            self._items[key] = value
            return value
        value = load_mha_array(path)
        if self.max_items:
            self._items[key] = value
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
        return value


@lru_cache(maxsize=256)
def _geometry(path_text: str) -> SpatialGeometry:
    return SpatialGeometry.from_mha_header(read_mha_header(Path(path_text)))


def _mri_bounds(image: NDArray[np.floating]) -> tuple[float, float]:
    positive = np.asarray(image, dtype=np.float32)
    positive = positive[positive > 0]
    if positive.size == 0:
        return (0.0, 1.0)
    low, high = np.percentile(positive, (0.5, 99.5))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


class ProtonPatchDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: Path,
        splits_path: Path,
        split: str,
        *,
        modality: str,
        patch_size_zyx: tuple[int, int, int] = (96, 96, 96),
        dose_scale: float = 1.0e-4,
        positive_patch_probability: float = 0.9,
        cache_size: int = 2,
        seed: int = 2026,
        deterministic_sampling: bool = False,
        augment: bool = False,
        aug_energy_jitter: float = 0.02,
        aug_density_scale: float = 0.03,
        aug_noise_std: float = 0.02,
        include_range_channels: bool = False,
        synthetic_ct_name: str | None = None,
    ):
        if modality not in {"ct", "mri"}:
            raise ValueError("modality must be ct or mri")
        if dose_scale <= 0:
            raise ValueError("dose_scale must be positive")
        patients = load_split_patients(splits_path, split)
        self.records = [record for record in read_manifest(manifest_path) if record.patient_id in patients]
        if not self.records:
            raise ValueError(f"no rows for split {split!r}")
        self.modality = modality
        self.patch_size_zyx = tuple(int(value) for value in patch_size_zyx)
        self.dose_scale = float(dose_scale)
        self.positive_patch_probability = float(positive_patch_probability)
        self.seed = int(seed)
        self.deterministic_sampling = bool(deterministic_sampling)
        self.augment = bool(augment) and split == "train"
        self.aug_energy_jitter = float(aug_energy_jitter)
        self.aug_density_scale = float(aug_density_scale)
        self.aug_noise_std = float(aug_noise_std)
        self.include_range_channels = bool(include_range_channels)
        # MRI carries no density. When a synthetic CT has been written beside
        # each MRI it is read as a second volume and becomes the source for the
        # density channel and for the water-equivalent depth, which is what
        # tells the network where a beamlet stops.
        self.synthetic_ct_name = synthetic_ct_name or None
        self._image_cache = _VolumeCache(cache_size)
        self._synthetic_cache = _VolumeCache(cache_size)
        self._rngs: dict[int, random.Random] = {}
        self._bounds: dict[str, tuple[float, float]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        del epoch

    def _rng(self, index: int) -> random.Random:
        if self.deterministic_sampling:
            return random.Random(self.seed + 104_729 * index)
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else -1
        if worker_id not in self._rngs:
            self._rngs[worker_id] = random.Random(int(worker.seed) if worker else self.seed)
        return self._rngs[worker_id]

    def _center(self, dose: NDArray[np.floating], rng: random.Random) -> tuple[int, int, int]:
        if rng.random() < self.positive_patch_probability and float(dose.max()) > 0:
            threshold = 0.1 * float(dose.max())
            for _ in range(256):
                point = tuple(rng.randrange(size) for size in dose.shape)
                if float(dose[point]) >= threshold:
                    return point
            return tuple(int(value) for value in np.unravel_index(int(np.argmax(dose)), dose.shape))
        return tuple(rng.randrange(size) for size in dose.shape)

    def _apply_augmentation(
        self,
        image_patch: NDArray[np.float32],
        dose_patch: NDArray[np.float32],
        rng: random.Random,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if self.aug_density_scale > 0:
            scale = 1.0 + rng.gauss(0, self.aug_density_scale)
            dose_patch = dose_patch * np.float32(scale)

        if self.aug_noise_std > 0:
            if self.modality == "ct":
                body = image_patch > -500
            else:
                body = image_patch > 0.01
            if body.any():
                noise = np.random.default_rng(rng.getrandbits(64)).normal(
                    0, self.aug_noise_std, image_patch.shape
                ).astype(np.float32)
                intensity_range = np.abs(image_patch[body]).max()
                if intensity_range > 0:
                    image_patch[body] += noise[body] * intensity_range

        return image_patch, dose_patch

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = record.ct_path if self.modality == "ct" else record.mr_path
        image = np.asarray(self._image_cache.get(image_path), dtype=np.float32)
        dose = np.asarray(load_mha_array(record.dose_path), dtype=np.float32)
        if image.shape != dose.shape:
            raise ValueError(f"image/dose mismatch for {record.patient_id}: {image.shape} != {dose.shape}")
        center = np.asarray(self._center(dose, self._rng(index)), dtype=np.int64)
        patch_size = np.asarray(self.patch_size_zyx, dtype=np.int64)
        start = center - patch_size // 2
        pad = -1024.0 if self.modality == "ct" else 0.0
        image_patch = crop_with_padding(image, start, patch_size, pad_value=pad)
        dose_patch = crop_with_padding(dose, start, patch_size, pad_value=0.0)

        if self.augment:
            image_patch, dose_patch = self._apply_augmentation(
                image_patch, dose_patch, self._rng(index)
            )

        synthetic = None
        synthetic_patch = None
        if self.synthetic_ct_name and self.modality == "mri":
            synthetic_path = image_path.parent / self.synthetic_ct_name
            synthetic = np.asarray(self._synthetic_cache.get(synthetic_path), dtype=np.float32)
            if synthetic.shape != image.shape:
                raise ValueError(
                    f"synthetic CT/MRI mismatch for {record.patient_id}: "
                    f"{synthetic.shape} != {image.shape}"
                )
            synthetic_patch = crop_with_padding(
                synthetic, start, patch_size, pad_value=-1000.0
            )

        bounds = None
        if self.modality == "mri":
            key = str(image_path.resolve())
            if key not in self._bounds:
                self._bounds[key] = _mri_bounds(image)
            bounds = self._bounds[key]
        geometry = _geometry(str(image_path.resolve()))
        wepl_patch = None
        if self.include_range_channels:
            # Depth integrates the unaugmented image along each ray, and slices
            # are independent along z, so only the patch's z-slab is traced —
            # but it must span the full transverse extent the rays cross.
            z_begin = max(int(start[0]), 0)
            z_end = min(int(start[0] + patch_size[0]), image.shape[0])
            if z_end > z_begin:
                wepl_slab = compute_proton_wepl(
                    image[z_begin:z_end],
                    geometry,
                    record.condition.gantry_angle_deg,
                    modality=self.modality,
                    intensity_bounds=bounds,
                    synthetic_hu=(
                        synthetic[z_begin:z_end] if synthetic is not None else None
                    ),
                )
                wepl_patch = crop_with_padding(
                    wepl_slab,
                    (int(start[0]) - z_begin, int(start[1]), int(start[2])),
                    patch_size,
                    pad_value=0.0,
                )
            else:
                wepl_patch = np.zeros(tuple(int(v) for v in patch_size), dtype=np.float32)
        channels = build_proton_channels(
            image_patch,
            start,
            geometry,
            record.condition,
            modality=self.modality,
            intensity_bounds=bounds,
            wepl_patch=wepl_patch,
            synthetic_hu_patch=synthetic_patch,
        )
        return {
            "input": torch.from_numpy(np.ascontiguousarray(channels)),
            "target": torch.from_numpy(np.ascontiguousarray(dose_patch[None] / self.dose_scale)),
            "target_max": torch.tensor(float(dose.max()) / self.dose_scale, dtype=torch.float32),
            "gantry_angle_deg": torch.tensor(record.condition.gantry_angle_deg, dtype=torch.float32),
            "patient_id": record.patient_id,
            "beam_idx": record.beam_idx,
            "ray_idx": record.ray_idx,
            "beamlet_idx": record.beamlet_idx,
        }
