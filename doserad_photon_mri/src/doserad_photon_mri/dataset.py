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

from .conditioning import (
    PhotonCondition,
    SpatialGeometry,
    build_condition_channels,
    hu_to_mass_density,
    mri_foreground_bounds,
)
from .mha import load_mha_array
from .dataset_index import read_mha_header


@dataclass(frozen=True)
class ManifestRecord:
    patient_id: str
    anatomy_group: str
    image_path: Path
    metadata_path: Path
    dose_path: Path
    beam_idx: int
    cp_idx: int


def read_manifest(path: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                ManifestRecord(
                    patient_id=row["patient_id"],
                    anatomy_group=row["anatomy_group"],
                    image_path=Path(row["image_path"]),
                    metadata_path=Path(row["metadata_path"]),
                    dose_path=Path(row["dose_path"]),
                    beam_idx=int(row["beam_idx"]),
                    cp_idx=int(row["cp_idx"]),
                )
            )
    return records


def load_split_patients(path: Path, split: str) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if split not in payload or not isinstance(payload[split], list):
        raise ValueError(f"split file has no list named {split!r}")
    return {str(patient_id) for patient_id in payload[split]}


@lru_cache(maxsize=128)
def _load_plan(path_text: str) -> dict[str, Any]:
    with Path(path_text).open(encoding="utf-8") as handle:
        return json.load(handle)


def lookup_condition(record: ManifestRecord) -> PhotonCondition:
    plan = _load_plan(str(record.metadata_path.resolve()))
    matching_beams = [beam for beam in plan["beams"] if int(beam["beam_idx"]) == record.beam_idx]
    if len(matching_beams) != 1:
        raise ValueError(f"cannot uniquely find beam {record.beam_idx} for {record.patient_id}")
    beam = matching_beams[0]
    matching_cps = [
        cp for cp in beam["control_points"] if int(cp["cp_idx"]) == record.cp_idx
    ]
    if len(matching_cps) != 1:
        raise ValueError(
            f"cannot uniquely find beam {record.beam_idx}, CP {record.cp_idx} "
            f"for {record.patient_id}"
        )
    return PhotonCondition.from_json(beam, matching_cps[0])


def crop_with_padding(
    array: NDArray[np.floating],
    start_zyx: Sequence[int],
    patch_size_zyx: Sequence[int],
    pad_value: float,
) -> NDArray[np.float32]:
    """Crop a 3D patch; negative/out-of-range coordinates are padded."""
    start = np.asarray(start_zyx, dtype=np.int64)
    patch_size = np.asarray(patch_size_zyx, dtype=np.int64)
    end = start + patch_size
    source_start = np.maximum(start, 0)
    source_end = np.minimum(end, np.asarray(array.shape))
    destination_start = source_start - start
    destination_end = destination_start + (source_end - source_start)

    output = np.full(tuple(int(value) for value in patch_size), pad_value, dtype=np.float32)
    if np.all(source_end > source_start):
        output[
            destination_start[0] : destination_end[0],
            destination_start[1] : destination_end[1],
            destination_start[2] : destination_end[2],
        ] = array[
            source_start[0] : source_end[0],
            source_start[1] : source_end[1],
            source_start[2] : source_end[2],
        ]
    return output


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


class PhotonMRIPatchDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: Path,
        splits_path: Path,
        split: str,
        *,
        patch_size_zyx: tuple[int, int, int] = (96, 96, 96),
        dose_scale: float = 1.0e-4,
        positive_patch_probability: float = 0.8,
        positive_threshold_fraction: float = 0.1,
        image_cache_size: int = 2,
        seed: int = 2026,
        deterministic_sampling: bool = False,
        include_physics_priors: bool = False,
        include_density_target: bool = False,
        ct_cache_size: int = 1,
        augment: bool = False,
        aug_intensity_jitter: float = 0.05,
        aug_density_scale: float = 0.05,
        aug_noise_std: float = 0.02,
        aug_flip_si: bool = True,
    ):
        if dose_scale <= 0:
            raise ValueError("dose_scale must be positive")
        if any(size <= 0 for size in patch_size_zyx):
            raise ValueError("patch dimensions must be positive")
        if not 0 <= positive_patch_probability <= 1:
            raise ValueError("positive_patch_probability must be between 0 and 1")

        split_patients = load_split_patients(splits_path, split)
        all_records = read_manifest(manifest_path)
        self.records = [record for record in all_records if record.patient_id in split_patients]
        if not self.records:
            raise ValueError(f"no manifest rows found for split {split!r}")

        self.split = split
        self.patch_size_zyx = tuple(int(value) for value in patch_size_zyx)
        self.dose_scale = float(dose_scale)
        self.positive_patch_probability = float(positive_patch_probability)
        self.positive_threshold_fraction = float(positive_threshold_fraction)
        self.seed = int(seed)
        self.deterministic_sampling = bool(deterministic_sampling)
        self.include_physics_priors = bool(include_physics_priors)
        self.include_density_target = bool(include_density_target)
        self.augment = bool(augment) and split == "train"
        self.aug_intensity_jitter = float(aug_intensity_jitter)
        self.aug_density_scale = float(aug_density_scale)
        self.aug_noise_std = float(aug_noise_std)
        self.aug_flip_si = bool(aug_flip_si)
        self.epoch = 0
        self._image_cache = _VolumeCache(image_cache_size)
        self._ct_cache = _VolumeCache(ct_cache_size)
        self._random_generators: dict[int, random.Random] = {}

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, index: int) -> random.Random:
        if self.deterministic_sampling:
            # Validation must present exactly the same crop for a record on
            # every epoch, independent of worker assignment and process state.
            return random.Random(self.seed + 104_729 * int(index))
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else -1
        if worker_id not in self._random_generators:
            worker_seed = int(worker.seed) if worker is not None else self.seed
            self._random_generators[worker_id] = random.Random(worker_seed)
        return self._random_generators[worker_id]

    def _choose_center(
        self, dose: NDArray[np.floating], rng: random.Random
    ) -> tuple[int, int, int]:
        use_positive = rng.random() < self.positive_patch_probability and float(dose.max()) > 0
        if use_positive:
            threshold = self.positive_threshold_fraction * float(dose.max())
            for _ in range(128):
                candidate = tuple(rng.randrange(size) for size in dose.shape)
                if float(dose[candidate]) >= threshold:
                    return candidate
            maximum = np.unravel_index(int(np.argmax(dose)), dose.shape)
            return tuple(int(value) for value in maximum)
        return tuple(rng.randrange(size) for size in dose.shape)

    def _apply_augmentation(
        self,
        image_patch: NDArray[np.float32],
        dose_patch: NDArray[np.float32],
        rng: random.Random,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if rng.random() < 0.5 and self.aug_flip_si:
            image_patch = np.ascontiguousarray(image_patch[::-1])
            dose_patch = np.ascontiguousarray(dose_patch[::-1])

        if self.aug_intensity_jitter > 0:
            foreground = image_patch > 0.01
            if foreground.any():
                scale = 1.0 + rng.gauss(0, self.aug_intensity_jitter)
                image_patch[foreground] *= np.float32(scale)

        if self.aug_density_scale > 0:
            scale = 1.0 + rng.gauss(0, self.aug_density_scale)
            dose_patch = dose_patch * np.float32(scale)

        if self.aug_noise_std > 0:
            foreground = image_patch > 0.01
            if foreground.any():
                noise = np.random.default_rng(rng.getrandbits(64)).normal(
                    0, self.aug_noise_std, image_patch.shape
                ).astype(np.float32)
                intensity_scale = np.abs(image_patch).max()
                if intensity_scale > 0:
                    image_patch[foreground] += noise[foreground] * intensity_scale

        return image_patch, dose_patch

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        rng = self._rng(index)
        dose = np.asarray(load_mha_array(record.dose_path), dtype=np.float32)
        image = np.asarray(self._image_cache.get(record.image_path), dtype=np.float32)
        if dose.shape != image.shape:
            raise ValueError(
                f"MRI/dose shape mismatch for {record.patient_id}: {image.shape} != {dose.shape}"
            )

        center = np.asarray(self._choose_center(dose, rng), dtype=np.int64)
        patch_size = np.asarray(self.patch_size_zyx, dtype=np.int64)
        start = center - patch_size // 2
        image_patch = crop_with_padding(image, start, patch_size, pad_value=0.0)
        dose_patch = crop_with_padding(dose, start, patch_size, pad_value=0.0)

        if self.augment:
            image_patch, dose_patch = self._apply_augmentation(image_patch, dose_patch, rng)

        header = read_mha_header(record.image_path)
        geometry = SpatialGeometry.from_mha_header(header)
        condition = lookup_condition(record)
        input_channels = build_condition_channels(
            image_patch,
            patch_start_zyx=start,
            geometry=geometry,
            condition=condition,
            intensity_bounds=mri_foreground_bounds(image),
            include_physics_priors=self.include_physics_priors,
        )
        sample = {
            "input": torch.from_numpy(np.ascontiguousarray(input_channels)),
            "target": torch.from_numpy(np.ascontiguousarray(dose_patch[None] / self.dose_scale)),
            "patient_id": record.patient_id,
            "beam_idx": record.beam_idx,
            "cp_idx": record.cp_idx,
            "gantry_angle_deg": torch.as_tensor(
                condition.gantry_angle_deg, dtype=torch.float32
            ),
            "dose_scale": self.dose_scale,
            "target_max": torch.as_tensor(
                float(dose.max()) / self.dose_scale, dtype=torch.float32
            ),
            "patch_start_zyx": torch.as_tensor(start, dtype=torch.int64),
        }
        if self.include_density_target:
            ct_path = record.image_path.with_name("ct.mha")
            if not ct_path.is_file():
                raise FileNotFoundError(f"paired CT is required for MRI supervision: {ct_path}")
            ct = np.asarray(self._ct_cache.get(ct_path), dtype=np.float32)
            if ct.shape != image.shape:
                raise ValueError(
                    f"paired CT/MRI shape mismatch for {record.patient_id}: "
                    f"{ct.shape} != {image.shape}"
                )
            ct_patch = crop_with_padding(ct, start, patch_size, pad_value=-1024.0)
            density = hu_to_mass_density(ct_patch)
            sample["density_target"] = torch.from_numpy(
                np.ascontiguousarray(density[None])
            )
            sample["density_mask"] = torch.from_numpy(
                np.ascontiguousarray((ct_patch > -1000.0)[None])
            )
        return sample
