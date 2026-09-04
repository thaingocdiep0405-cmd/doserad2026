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
)
from .mha import load_mha_array
from .dataset_index import read_mha_header
from .radiological import compute_radiological_depth


@dataclass(frozen=True)
class ManifestRecord:
    patient_id: str
    anatomy_group: str
    ct_path: Path
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
                    ct_path=Path(row["ct_path"]),
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


class PhotonCTPatchDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: Path,
        splits_path: Path,
        split: str,
        *,
        patch_size_zyx: tuple[int, int, int] = (96, 96, 96),
        dose_scale: float = 1.0e-4,
        ct_clip: tuple[float, float] = (-1024.0, 2000.0),
        positive_patch_probability: float = 0.8,
        positive_threshold_fraction: float = 0.1,
        ct_cache_size: int = 2,
        seed: int = 2026,
        deterministic_sampling: bool = False,
        include_density: bool = False,
        include_physics_priors: bool = False,
        include_radiological_depth: bool = False,
        pb_dose_dir: Path | None = None,
        augment: bool = False,
        aug_hu_jitter: float = 50.0,
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
        self.ct_clip = ct_clip
        self.positive_patch_probability = float(positive_patch_probability)
        self.positive_threshold_fraction = float(positive_threshold_fraction)
        self.seed = int(seed)
        self.deterministic_sampling = bool(deterministic_sampling)
        self.include_radiological_depth = bool(include_radiological_depth)
        self.include_physics_priors = bool(
            include_physics_priors or include_radiological_depth
        )
        self.include_density = bool(include_density or self.include_physics_priors)
        self.pb_dose_dir = Path(pb_dose_dir) if pb_dose_dir is not None else None
        self.augment = bool(augment) and split == "train"
        self.aug_hu_jitter = float(aug_hu_jitter)
        self.aug_density_scale = float(aug_density_scale)
        self.aug_noise_std = float(aug_noise_std)
        self.aug_flip_si = bool(aug_flip_si)
        self.epoch = 0
        self._ct_cache = _VolumeCache(ct_cache_size)
        self._density_cache: OrderedDict[str, NDArray[np.float32]] = OrderedDict()
        self._density_cache_size = max(0, int(ct_cache_size))
        self._random_generators: dict[int, random.Random] = {}

    def _mass_density(self, ct_path: Path, ct: NDArray[np.float32]) -> NDArray[np.float32]:
        key = str(ct_path.resolve())
        if key in self._density_cache:
            value = self._density_cache.pop(key)
            self._density_cache[key] = value
            return value
        value = hu_to_mass_density(ct)
        if self._density_cache_size:
            self._density_cache[key] = value
            while len(self._density_cache) > self._density_cache_size:
                self._density_cache.popitem(last=False)
        return value

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

    def _apply_augmentation(
        self,
        ct_patch: NDArray[np.float32],
        dose_patch: NDArray[np.float32],
        rng: random.Random,
        flip_si: bool,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if flip_si:
            ct_patch = np.ascontiguousarray(ct_patch[::-1])
            dose_patch = np.ascontiguousarray(dose_patch[::-1])
        if self.aug_hu_jitter > 0:
            jitter = rng.gauss(0.0, self.aug_hu_jitter)
            body = ct_patch > -1000.0
            ct_patch = ct_patch.copy()
            ct_patch[body] += jitter
        if self.aug_density_scale > 0:
            scale = 1.0 + rng.gauss(0.0, self.aug_density_scale)
            dose_patch = dose_patch * scale
        if self.aug_noise_std > 0:
            noise = np.random.default_rng(rng.randint(0, 2**31)).normal(
                0.0, self.aug_noise_std, ct_patch.shape
            ).astype(np.float32)
            body = ct_patch > -1000.0
            ct_patch = ct_patch.copy() if not ct_patch.flags.owndata else ct_patch
            ct_patch[body] += noise[body] * (ct_patch[body] - self.ct_clip[0])
        return ct_patch, dose_patch

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

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        rng = self._rng(index)
        dose = np.asarray(load_mha_array(record.dose_path), dtype=np.float32)
        ct = np.asarray(self._ct_cache.get(record.ct_path), dtype=np.float32)
        if dose.shape != ct.shape:
            raise ValueError(
                f"CT/dose shape mismatch for {record.patient_id}: {ct.shape} != {dose.shape}"
            )

        center = np.asarray(self._choose_center(dose, rng), dtype=np.int64)
        patch_size = np.asarray(self.patch_size_zyx, dtype=np.int64)
        start = center - patch_size // 2
        ct_patch = crop_with_padding(ct, start, patch_size, pad_value=-1024.0)
        dose_patch = crop_with_padding(dose, start, patch_size, pad_value=0.0)

        flip_si = bool(
            self.augment and self.aug_flip_si and rng.random() < 0.5
        )
        if self.augment:
            ct_patch, dose_patch = self._apply_augmentation(
                ct_patch, dose_patch, rng, flip_si
            )

        header = read_mha_header(record.ct_path)
        geometry = SpatialGeometry.from_mha_header(header)
        condition = lookup_condition(record)

        radiological_patch = None
        if self.include_radiological_depth:
            # Depth integrates the unaugmented CT along each ray; HU jitter
            # and noise on the CT patch are treated as input perturbations
            # instead of being propagated into the ray sums. Slices are
            # independent along z, so only the patch's z-slab is traced.
            density = self._mass_density(record.ct_path, ct)
            z_begin = max(int(start[0]), 0)
            z_end = min(int(start[0] + patch_size[0]), ct.shape[0])
            if z_end > z_begin:
                depth_slab = compute_radiological_depth(
                    density[z_begin:z_end],
                    geometry.spacing_xyz,
                    condition.gantry_angle_deg,
                )
                radiological_patch = crop_with_padding(
                    depth_slab,
                    (int(start[0]) - z_begin, int(start[1]), int(start[2])),
                    patch_size,
                    pad_value=0.0,
                )
            else:
                radiological_patch = np.zeros(
                    tuple(int(value) for value in patch_size), dtype=np.float32
                )
            if flip_si:
                radiological_patch = np.ascontiguousarray(radiological_patch[::-1])

        pb_patch = None
        if self.pb_dose_dir is not None:
            pb_path = (
                self.pb_dose_dir
                / f"{record.patient_id}_B{record.beam_idx}_CP{record.cp_idx:03d}.npz"
            )
            if pb_path.exists():
                pb_volume = np.load(pb_path)["dose"].astype(np.float32)
                if pb_volume.shape == ct.shape:
                    pb_patch = crop_with_padding(
                        pb_volume, start, patch_size, pad_value=0.0
                    )
                    if flip_si:
                        pb_patch = np.ascontiguousarray(pb_patch[::-1])
        has_pb = pb_patch is not None
        if pb_patch is None:
            pb_patch = np.zeros(
                tuple(int(value) for value in patch_size), dtype=np.float32
            )

        input_channels = build_condition_channels(
            ct_patch,
            patch_start_zyx=start,
            geometry=geometry,
            condition=condition,
            ct_clip=self.ct_clip,
            include_density=self.include_density,
            include_physics_priors=self.include_physics_priors,
            radiological_depth_patch=radiological_patch,
        )

        return {
            "pb_target": torch.from_numpy(
                np.ascontiguousarray(pb_patch[None] / self.dose_scale)
            ),
            "has_pb": torch.as_tensor(has_pb, dtype=torch.bool),
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
