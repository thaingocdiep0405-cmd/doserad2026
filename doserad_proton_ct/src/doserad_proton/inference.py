from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from doserad_photon_ct.conditioning import SpatialGeometry
from doserad_photon_ct.dataset import crop_with_padding
from doserad_photon_ct.dataset_index import read_mha_header
from doserad_photon_ct.inference import (
    _accumulate_patch,
    _patch_slices,
    gaussian_blend_weight,
    sliding_window_starts,
)
from doserad_photon_ct.mha import load_mha_array

from .conditioning import (
    BRAGG_EXPONENT,
    RANGE_ALPHA_MM,
    RANGE_EXPONENT,
    STRAGGLING_FRACTION,
    WEPL_NORMALIZATION_MM,
    PreparedProtonPatch,
    ProtonCondition,
    build_proton_channels,
    build_proton_channels_batch,
    build_range_channels,
    compute_proton_wepl,
    prepare_proton_patch,
)
from .data import ProtonRecord, _mri_bounds


@dataclass(frozen=True)
class _TorchProtonPatch:
    normalized: torch.Tensor
    body: torch.Tensor
    density: torch.Tensor
    px: torch.Tensor
    py: torch.Tensor
    pz: torch.Tensor
    wepl: torch.Tensor | None = None


def _to_torch_patch(
    prepared: PreparedProtonPatch,
    device: torch.device,
    wepl_patch: NDArray[np.floating] | None = None,
) -> _TorchProtonPatch:
    return _TorchProtonPatch(
        wepl=None if wepl_patch is None else torch.as_tensor(
            np.ascontiguousarray(wepl_patch, dtype=np.float32), device=device
        ),
        normalized=torch.as_tensor(prepared.normalized, device=device),
        body=torch.as_tensor(prepared.body, device=device),
        density=torch.as_tensor(prepared.density, device=device),
        px=torch.as_tensor(prepared.px, device=device),
        py=torch.as_tensor(prepared.py, device=device),
        pz=torch.as_tensor(prepared.pz, device=device),
    )


def _build_proton_channels_torch(
    prepared: _TorchProtonPatch,
    conditions: Sequence[ProtonCondition],
) -> torch.Tensor:
    """Vectorized GPU equivalent of the NumPy batch channel builder."""
    device = prepared.normalized.device
    source = torch.tensor(
        [condition.ray_source_xyz for condition in conditions],
        dtype=torch.float32,
        device=device,
    )
    target = torch.tensor(
        [condition.ray_target_xyz for condition in conditions],
        dtype=torch.float32,
        device=device,
    )
    direction = target - source
    norm = torch.linalg.vector_norm(direction, dim=1)
    if bool(torch.any(norm <= 0).item()):
        raise ValueError("ray source and target must differ")
    direction = direction / norm[:, None]

    px = prepared.px.unsqueeze(0)
    py = prepared.py.unsqueeze(0)
    pz = prepared.pz.unsqueeze(0)
    rel_source_x = px - source[:, 0, None, None, None]
    rel_source_y = py - source[:, 1, None, None, None]
    rel_source_z = pz - source[:, 2, None, None, None]
    along = (
        rel_source_x * direction[:, 0, None, None, None]
        + rel_source_y * direction[:, 1, None, None, None]
        + rel_source_z * direction[:, 2, None, None, None]
    )
    closest_x = source[:, 0, None, None, None] + along * direction[:, 0, None, None, None]
    closest_y = source[:, 1, None, None, None] + along * direction[:, 1, None, None, None]
    closest_z = source[:, 2, None, None, None] + along * direction[:, 2, None, None, None]
    radial_sq = (px - closest_x).square() + (py - closest_y).square() + (pz - closest_z).square()
    sigma = torch.tensor(
        [max(float(condition.sigma_spot_mm), 1.0) for condition in conditions],
        dtype=torch.float32,
        device=device,
    )
    fluence = torch.exp(-0.5 * radial_sq / sigma[:, None, None, None].square())
    fluence = fluence * (along > 0)

    rel_target_x = px - target[:, 0, None, None, None]
    rel_target_y = py - target[:, 1, None, None, None]
    rel_target_z = pz - target[:, 2, None, None, None]
    depth = (
        rel_target_x * direction[:, 0, None, None, None]
        + rel_target_y * direction[:, 1, None, None, None]
        + rel_target_z * direction[:, 2, None, None, None]
    )
    lateral = (
        rel_target_x * -direction[:, 1, None, None, None]
        + rel_target_y * direction[:, 0, None, None, None]
    )
    target_prior = torch.exp(
        -0.5 * (rel_target_x.square() + rel_target_y.square() + rel_target_z.square()) / (50.0**2)
    )
    energy = torch.tensor(
        [condition.energy_mev for condition in conditions],
        dtype=torch.float32,
        device=device,
    )
    energy = (2.0 * (energy - 31.7290) / (200.7966 - 31.7290) - 1.0).clamp(-1.0, 1.0)
    energy_width = torch.tensor(
        [condition.sigma_energy_mev / 7.0 for condition in conditions],
        dtype=torch.float32,
        device=device,
    ).clamp(0.0, 1.0)

    shape = tuple(int(value) for value in prepared.normalized.shape)
    count = 10 if prepared.wepl is None else 12
    channels = torch.empty((len(conditions), count, *shape), dtype=torch.float32, device=device)
    channels[:, 0] = prepared.normalized
    channels[:, 1] = prepared.body
    channels[:, 2] = fluence
    channels[:, 3] = (depth / 400.0).clamp(-1.0, 1.0)
    channels[:, 4] = (lateral / 200.0).clamp(-1.0, 1.0)
    channels[:, 5] = (rel_target_z / 200.0).clamp(-1.0, 1.0)
    channels[:, 6] = energy[:, None, None, None]
    channels[:, 7] = energy_width[:, None, None, None]
    channels[:, 8] = target_prior
    channels[:, 9] = prepared.density
    if prepared.wepl is not None:
        # Same Bortfeld prior as the NumPy builder, evaluated per beamlet
        # because the range depends on the beamlet energy.
        energies = torch.tensor(
            [condition.energy_mev for condition in conditions],
            dtype=torch.float32,
            device=device,
        )
        ranges = (RANGE_ALPHA_MM * energies.pow(RANGE_EXPONENT))[:, None, None, None]
        residual = ranges - prepared.wepl.unsqueeze(0)
        straggling = (STRAGGLING_FRACTION * ranges).clamp_min(1.0)
        falloff = 0.5 * (1.0 + torch.tanh(residual / straggling))
        prior = falloff * residual.clamp_min(1.0).pow(BRAGG_EXPONENT)
        channels[:, 10] = (prepared.wepl.unsqueeze(0) / WEPL_NORMALIZATION_MM).clamp(0.0, 2.0)
        channels[:, 11] = prior.clamp(0.0, 1.0)
    return channels


@torch.inference_mode()
def predict_record_volume(
    model: torch.nn.Module,
    record: ProtonRecord,
    *,
    modality: str,
    device: torch.device,
    patch_size_zyx: tuple[int, int, int] = (128, 128, 128),
    dose_scale: float,
    overlap: float = 0.25,
    batch_size: int = 4,
    amp: bool = True,
    skip_empty_ray: bool = True,
    mask_outside_body: bool = True,
    relative_cutoff: float = 1.0e-4,
    ray_gate_threshold: float = 0.0,
) -> tuple[NDArray[np.float32], float]:
    """Predict one complete proton pencil-beam dose on its native grid."""
    if modality not in {"ct", "mri"}:
        raise ValueError("modality must be ct or mri")
    image_path = record.ct_path if modality == "ct" else record.mr_path
    image = np.asarray(load_mha_array(image_path), dtype=np.float32)
    geometry = SpatialGeometry.from_mha_header(read_mha_header(image_path))
    bounds = _mri_bounds(image) if modality == "mri" else None
    pad_value = -1024.0 if modality == "ct" else 0.0
    starts = [
        (z, y, x)
        for z in sliding_window_starts(image.shape[0], patch_size_zyx[0], overlap)
        for y in sliding_window_starts(image.shape[1], patch_size_zyx[1], overlap)
        for x in sliding_window_starts(image.shape[2], patch_size_zyx[2], overlap)
    ]
    output = np.zeros(image.shape, dtype=np.float32)
    total_weight = np.zeros(image.shape, dtype=np.float32)
    weight = gaussian_blend_weight(patch_size_zyx)
    model.eval()
    started = time.perf_counter()

    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        batch_inputs: list[NDArray[np.float32]] = []
        active_starts: list[tuple[int, int, int]] = []
        for start in batch_starts:
            image_patch = crop_with_padding(image, start, patch_size_zyx, pad_value=pad_value)
            channels = build_proton_channels(
                image_patch,
                start,
                geometry,
                record.condition,
                modality=modality,
                intensity_bounds=bounds,
            )
            if skip_empty_ray and float(channels[2].max()) < 1.0e-6:
                continue
            batch_inputs.append(channels)
            active_starts.append(start)
        if not batch_inputs:
            continue
        inputs = torch.from_numpy(np.stack(batch_inputs)).to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            predictions = model(inputs)[:, 0]
            if ray_gate_threshold > 0.0:
                ray_gate = inputs[:, 2] >= ray_gate_threshold
                predictions = predictions * ray_gate
        predictions_np = predictions.float().cpu().numpy() * float(dose_scale)
        for prediction, start in zip(predictions_np, active_starts):
            _accumulate_patch(output, total_weight, prediction, weight, start)

    prediction = output / np.maximum(total_weight, 1.0e-8)
    if mask_outside_body:
        if modality == "ct":
            prediction[image <= -1000.0] = 0.0
        else:
            low = bounds[0] if bounds else 0.0
            prediction[image <= low] = 0.0
    peak = float(prediction.max())
    if relative_cutoff > 0.0 and peak > 0.0:
        prediction[prediction < relative_cutoff * peak] = 0.0
    return np.asarray(prediction, dtype=np.float32), time.perf_counter() - started


def _window_condition_activity(
    starts: Sequence[tuple[int, int, int]],
    patch_size_zyx: tuple[int, int, int],
    geometry: SpatialGeometry,
    conditions: Sequence[ProtonCondition],
    *,
    gate_threshold: float,
    distal_margin_mm: float | None,
) -> NDArray[np.bool_]:
    """Which (window, beamlet) pairs can receive nonzero gated dose.

    The ray gate multiplies every prediction by ``fluence >= gate_threshold``,
    and fluence is a radial Gaussian around the beam half-line, so a window
    whose closest voxel lies beyond the gate radius contributes exactly zeros.
    The box test uses the window's bounding sphere, which keeps a superset of
    the windows the per-voxel fluence check would keep.
    """
    direction_matrix = np.asarray(geometry.direction, dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(geometry.spacing_xyz, dtype=np.float64)
    origin = np.asarray(geometry.origin_xyz, dtype=np.float64)
    half_index = 0.5 * (np.asarray(patch_size_zyx, dtype=np.float64)[::-1] - 1.0)
    half_diagonal = float(np.linalg.norm(half_index * spacing))
    centers = np.empty((len(starts), 3), dtype=np.float64)
    for row, (start_z, start_y, start_x) in enumerate(starts):
        index_xyz = np.asarray([start_x, start_y, start_z], dtype=np.float64) + half_index
        centers[row] = origin + direction_matrix @ (index_xyz * spacing)

    sources = np.asarray([c.ray_source_xyz for c in conditions], dtype=np.float64)
    targets = np.asarray([c.ray_target_xyz for c in conditions], dtype=np.float64)
    axes = targets - sources
    lengths = np.linalg.norm(axes, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("ray source and target must differ")
    axes /= lengths[:, None]
    sigmas = np.asarray([max(float(c.sigma_spot_mm), 1.0) for c in conditions])
    gate_radius = sigmas * float(np.sqrt(max(-2.0 * np.log(gate_threshold), 0.0)))

    rel = centers[:, None, :] - sources[None, :, :]
    along = np.einsum("wcd,cd->wc", rel, axes)
    closest = sources[None] + np.maximum(along, 0.0)[..., None] * axes[None]
    distance = np.linalg.norm(centers[:, None, :] - closest, axis=2)
    active = distance <= gate_radius[None, :] + half_diagonal
    if distal_margin_mm is not None:
        active &= (along - half_diagonal) <= (lengths[None, :] + float(distal_margin_mm))
    return active


def corridor_window_starts(
    image_shape_zyx: tuple[int, int, int],
    patch_size_zyx: tuple[int, int, int],
    geometry: SpatialGeometry,
    condition: ProtonCondition,
    *,
    gate_threshold: float,
    distal_margin_mm: float,
    overlap: float = 0.0,
    wepl_volume: NDArray[np.floating] | None = None,
    range_energy_mev: float | None = None,
) -> list[tuple[int, int, int]]:
    """Window starts covering only the gated corridor of one ray.

    The ray gate zeros every voxel whose fluence lies below ``gate_threshold``,
    so windows are laid over the axis-aligned bounding box of the corridor
    (half-line from source through target, radius from the gate threshold,
    ended ``distal_margin_mm`` past the target) instead of the full grid. For
    the in-plane beams of this challenge that shrinks the tiled volume by
    several times.
    """
    direction_matrix = np.asarray(geometry.direction, dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(geometry.spacing_xyz, dtype=np.float64)
    origin = np.asarray(geometry.origin_xyz, dtype=np.float64)
    source = np.asarray(condition.ray_source_xyz, dtype=np.float64)
    target = np.asarray(condition.ray_target_xyz, dtype=np.float64)
    axis = target - source
    length = float(np.linalg.norm(axis))
    if length <= 0:
        raise ValueError("ray source and target must differ")
    axis /= length
    sigma = max(float(condition.sigma_spot_mm), 1.0)
    radius = sigma * float(np.sqrt(max(-2.0 * np.log(gate_threshold), 0.0)))

    # Sample the segment from the source to just past the target, convert to
    # voxel indices and expand by the corridor radius per axis.
    samples = source[None, :] + np.linspace(
        0.0, length + float(distal_margin_mm), 256
    )[:, None] * axis[None, :]
    relative = (samples - origin[None, :]) @ direction_matrix  # inverse rotation
    index_xyz = relative / spacing[None, :]
    if wepl_volume is not None:
        # The spot position sits on the isocentre plane, not at the Bragg peak,
        # so the corridor is truncated where the accumulated water-equivalent
        # depth passes the beamlet range plus a few millimetres of straggling.
        # A ray carries several beamlets and the corridor must reach the
        # deepest of them, so the cut uses the highest energy in the group.
        energy = float(range_energy_mev or condition.energy_mev)
        range_mm = RANGE_ALPHA_MM * energy ** RANGE_EXPONENT
        limit = range_mm + max(3.0 * STRAGGLING_FRACTION * range_mm, 5.0)
        shape_zyx = np.asarray(wepl_volume.shape, dtype=int)
        index_zyx = np.rint(index_xyz[:, ::-1]).astype(int)
        inside = np.all((index_zyx >= 0) & (index_zyx < shape_zyx[None, :]), axis=1)
        depth_along = np.zeros(len(index_zyx), dtype=np.float32)
        if inside.any():
            rows = index_zyx[inside]
            depth_along[inside] = wepl_volume[rows[:, 0], rows[:, 1], rows[:, 2]]
        beyond = np.flatnonzero(depth_along > limit)
        if beyond.size:
            index_xyz = index_xyz[: int(beyond[0]) + 1]
    pad_xyz = radius / spacing
    low_xyz = np.floor(index_xyz.min(axis=0) - pad_xyz).astype(int)
    high_xyz = np.ceil(index_xyz.max(axis=0) + pad_xyz).astype(int)
    shape_xyz = np.asarray(image_shape_zyx[::-1], dtype=int)
    low_xyz = np.clip(low_xyz, 0, shape_xyz - 1)
    high_xyz = np.clip(high_xyz, 0, shape_xyz - 1)

    starts: list[tuple[int, int, int]] = []
    axes_starts = []
    for axis_index in range(3):  # z, y, x
        dim = int(image_shape_zyx[axis_index])
        patch = int(patch_size_zyx[axis_index])
        low = int(low_xyz[2 - axis_index])
        high = int(high_xyz[2 - axis_index])
        extent = high - low + 1
        if extent <= patch:
            anchor = max(0, min(low + (extent - patch) // 2, dim - patch))
            positions = [anchor] if dim > patch else [0]
        else:
            positions = [
                max(0, min(low + offset, dim - patch))
                for offset in sliding_window_starts(extent, patch, overlap)
            ]
        axes_starts.append(sorted(set(positions)))
    for z in axes_starts[0]:
        for y in axes_starts[1]:
            for x in axes_starts[2]:
                starts.append((z, y, x))
    return starts


@torch.inference_mode()
def predict_conditioned_arrays(
    model: torch.nn.Module,
    *,
    image: NDArray[np.floating],
    geometry: SpatialGeometry,
    conditions: Sequence[ProtonCondition],
    modality: str,
    device: torch.device,
    patch_size_zyx: tuple[int, int, int] = (128, 128, 128),
    dose_scale: float,
    overlap: float = 0.25,
    condition_batch_size: int = 4,
    amp: bool = True,
    skip_empty_ray: bool = True,
    mask_outside_body: bool = True,
    relative_cutoff: float = 1.0e-4,
    ray_gate_threshold: float = 0.0,
    pad_to_batch_size: bool = False,
    roi_mode: str = "off",
    roi_distal_margin_mm: float = 80.0,
    window_cache: dict | None = None,
    window_starts: Sequence[tuple[int, int, int]] | None = None,
    range_channels: bool = False,
    wepl_cache: dict | None = None,
    synthetic_hu: NDArray[np.floating] | None = None,
) -> list[NDArray[np.float32]]:
    """Predict several proton beamlets for one image with batched GPU forwards."""
    if not conditions:
        return []
    if modality not in {"ct", "mri"}:
        raise ValueError("modality must be ct or mri")
    if condition_batch_size < 1:
        raise ValueError("condition_batch_size must be positive")
    if roi_mode not in {"off", "corridor", "capsule", "bbox"}:
        raise ValueError("roi_mode must be off, corridor, capsule or bbox")

    if range_channels:
        angles = {condition.gantry_angle_deg for condition in conditions}
        if len(angles) > 1:
            # Water-equivalent depth is a property of the beam direction, so
            # each gantry angle is traced once over its own conditions.
            by_angle: dict[float, list[int]] = {}
            for index, condition in enumerate(conditions):
                by_angle.setdefault(condition.gantry_angle_deg, []).append(index)
            merged: list[NDArray[np.float32] | None] = [None] * len(conditions)
            for indices in by_angle.values():
                part = predict_conditioned_arrays(
                    model,
                    image=image,
                    geometry=geometry,
                    conditions=[conditions[i] for i in indices],
                    modality=modality,
                    device=device,
                    patch_size_zyx=patch_size_zyx,
                    dose_scale=dose_scale,
                    overlap=overlap,
                    condition_batch_size=condition_batch_size,
                    amp=amp,
                    skip_empty_ray=skip_empty_ray,
                    mask_outside_body=mask_outside_body,
                    relative_cutoff=relative_cutoff,
                    ray_gate_threshold=ray_gate_threshold,
                    pad_to_batch_size=pad_to_batch_size,
                    roi_mode=roi_mode,
                    roi_distal_margin_mm=roi_distal_margin_mm,
                    window_cache=window_cache,
                    window_starts=window_starts,
                    synthetic_hu=synthetic_hu,
                    range_channels=True,
                    wepl_cache=wepl_cache,
                )
                for target_index, prediction in zip(indices, part):
                    merged[target_index] = prediction
            return [np.asarray(item, dtype=np.float32) for item in merged]

    if roi_mode == "bbox":
        # Group beamlets by ray and tile windows only over each ray's gated
        # corridor. Groups get independent window grids (and therefore
        # independent blend weights), so each is delegated to a plain corridor
        # pass with its own starts.
        groups: dict[tuple, list[int]] = {}
        for index, condition in enumerate(conditions):
            key = (condition.ray_source_xyz, condition.ray_target_xyz)
            groups.setdefault(key, []).append(index)
        image_array = np.asarray(image, dtype=np.float32)
        bbox_wepl = None
        # Off by default: MRI carries no electron density, so the corridor is
        # traced through water. Beamlets crossing lung or air travel much
        # further than that estimate, and truncating at the water range then
        # cuts the Bragg peak away entirely (measured: one beamlet in ten loses
        # 99.9% of its dose) to save 12% of the inference time.
        if os.environ.get("BBOX_RANGE_CUT", "0") != "0":
            angle = conditions[0].gantry_angle_deg
            key = ("wepl", angle, modality)
            if wepl_cache is not None and key in wepl_cache:
                bbox_wepl = wepl_cache[key]
            else:
                bbox_wepl = compute_proton_wepl(
                    image_array,
                    geometry,
                    angle,
                    synthetic_hu=synthetic_hu,
                    modality=modality,
                    intensity_bounds=(
                        _mri_bounds(image_array) if modality == "mri" else None
                    ),
                    device=str(device) if device.type == "cuda" else None,
                )
                if wepl_cache is not None:
                    wepl_cache[key] = bbox_wepl
        results: list[NDArray[np.float32] | None] = [None] * len(conditions)
        for indices in groups.values():
            group_conditions = [conditions[i] for i in indices]
            starts_override = corridor_window_starts(
                tuple(int(v) for v in image_array.shape),
                patch_size_zyx,
                geometry,
                group_conditions[0],
                gate_threshold=max(ray_gate_threshold, 1.0e-6),
                distal_margin_mm=roi_distal_margin_mm,
                overlap=overlap,
                wepl_volume=bbox_wepl,
                range_energy_mev=max(c.energy_mev for c in group_conditions),
            )
            predictions = predict_conditioned_arrays(
                model,
                image=image_array,
                geometry=geometry,
                conditions=group_conditions,
                modality=modality,
                device=device,
                patch_size_zyx=patch_size_zyx,
                dose_scale=dose_scale,
                overlap=overlap,
                # A ray group is small (energies of one spot), so the batch is
                # the whole group; padding to the caller's batch size would
                # multiply the forward cost for nothing.
                condition_batch_size=min(condition_batch_size, len(group_conditions)),
                amp=amp,
                skip_empty_ray=skip_empty_ray,
                mask_outside_body=mask_outside_body,
                relative_cutoff=relative_cutoff,
                ray_gate_threshold=ray_gate_threshold,
                pad_to_batch_size=pad_to_batch_size,
                roi_mode="corridor",
                roi_distal_margin_mm=roi_distal_margin_mm,
                window_starts=starts_override,
                synthetic_hu=synthetic_hu,
                # Each ray group tiles its own corridor, so two groups almost
                # never share a window start and the cache never hits. What it
                # does do is hold seven 96^3 tensors per window alive on the
                # GPU, and that allocator pressure measured 12.8% slower than
                # simply rebuilding them (50.4 s against 44.7 s over 30 maps).
                # Windows are already built once per call and reused across the
                # beamlets of the group, which is the reuse that pays.
                window_cache=None,
                range_channels=range_channels,
                wepl_cache=wepl_cache,
            )
            for target_index, prediction in zip(indices, predictions):
                results[target_index] = prediction
        return [np.asarray(item, dtype=np.float32) for item in results]

    image = np.asarray(image, dtype=np.float32)
    bounds = _mri_bounds(image) if modality == "mri" else None
    pad_value = -1024.0 if modality == "ct" else 0.0
    if window_starts is not None:
        starts = [tuple(int(v) for v in start) for start in window_starts]
    else:
        starts = [
            (z, y, x)
            for z in sliding_window_starts(image.shape[0], patch_size_zyx[0], overlap)
            for y in sliding_window_starts(image.shape[1], patch_size_zyx[1], overlap)
            for x in sliding_window_starts(image.shape[2], patch_size_zyx[2], overlap)
        ]
    outputs = torch.zeros((len(conditions), *image.shape), dtype=torch.float32, device=device)
    total_weight = torch.zeros(image.shape, dtype=torch.float32, device=device)
    patch_weight = torch.from_numpy(gaussian_blend_weight(patch_size_zyx)).to(device)

    for start in starts:
        destination_slices, source_slices = _patch_slices(image.shape, patch_size_zyx, start)
        total_weight[destination_slices] += patch_weight[source_slices]

    wepl_volume = None
    if range_channels:
        angle = conditions[0].gantry_angle_deg
        cache_key = ("wepl", angle, modality)
        if wepl_cache is not None and cache_key in wepl_cache:
            wepl_volume = wepl_cache[cache_key]
        else:
            wepl_volume = compute_proton_wepl(
                image,
                geometry,
                angle,
                modality=modality,
                intensity_bounds=bounds,
                device=str(device) if device.type == "cuda" else None,
                synthetic_hu=synthetic_hu,
            )
            if wepl_cache is not None:
                wepl_cache[cache_key] = wepl_volume

    roi_activity = None
    if roi_mode != "off":
        roi_activity = _window_condition_activity(
            starts,
            patch_size_zyx,
            geometry,
            conditions,
            gate_threshold=max(ray_gate_threshold, 1.0e-6),
            distal_margin_mm=roi_distal_margin_mm if roi_mode == "capsule" else None,
        )

    model.eval()

    # One forward per (window, condition) pair leaves the GPU at batch one
    # whenever a ray carries a single beamlet, which is the normal case for the
    # graded data: bbox groups beamlets by ray and real plans have two per ray.
    # Patches are queued across windows instead and flushed as one batch, so the
    # batch size the caller asked for is actually reached. Every patch still
    # goes through the same forward with the same inputs, so the arithmetic is
    # unchanged; only how many of them ride in one call differs.
    pending_inputs: list[torch.Tensor] = []
    pending_rows: list[tuple[int, tuple[int, int, int]]] = []

    def flush_pending() -> None:
        if not pending_inputs:
            return
        batch = torch.cat(pending_inputs, dim=0) if len(pending_inputs) > 1 else pending_inputs[0]
        rows = list(pending_rows)
        pending_inputs.clear()
        pending_rows.clear()
        active_count = batch.shape[0]
        if pad_to_batch_size and active_count < condition_batch_size:
            padding = torch.zeros(
                (condition_batch_size - active_count, *batch.shape[1:]),
                dtype=batch.dtype,
                device=device,
            )
            batch = torch.cat((batch, padding), dim=0)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            batch_predictions = model(batch)[:active_count, 0]
            if ray_gate_threshold > 0.0:
                batch_predictions = batch_predictions * (
                    batch[:active_count, 2] >= ray_gate_threshold
                )
        predictions = batch_predictions.float() * float(dose_scale)
        for row, (condition_index, row_start) in enumerate(rows):
            destination_slices, source_slices = _patch_slices(
                image.shape, patch_size_zyx, row_start
            )
            outputs[(condition_index, *destination_slices)] += (
                predictions[(row, *source_slices)] * patch_weight[source_slices]
            )

    for start_index, start in enumerate(starts):
        if roi_activity is not None and not roi_activity[start_index].any():
            continue
        # Static per-window work (normalization, physical coordinates, density
        # and the GPU transfer) depends only on the image, so across chunked
        # calls for the same image it is reused instead of recomputed.
        cache_key = (start, patch_size_zyx, conditions[0].gantry_angle_deg)
        cached_entry = window_cache.get(cache_key) if window_cache is not None else None
        if cached_entry is not None:
            image_patch, prepared_patch, prepared_torch, wepl_patch = cached_entry
        else:
            image_patch = crop_with_padding(
                image, start, patch_size_zyx, pad_value=pad_value
            )
            wepl_patch = (
                None
                if wepl_volume is None
                else crop_with_padding(wepl_volume, start, patch_size_zyx, pad_value=0.0)
            )
            prepared_patch = prepare_proton_patch(
                image_patch,
                start,
                geometry,
                modality=modality,
                intensity_bounds=bounds,
                synthetic_hu_patch=(
                    None
                    if synthetic_hu is None
                    else crop_with_padding(
                        synthetic_hu, start, patch_size_zyx, pad_value=-1000.0
                    )
                ),
            )
            prepared_torch = (
                _to_torch_patch(prepared_patch, device, wepl_patch)
                if device.type == "cuda"
                else None
            )
            if window_cache is not None:
                window_cache[cache_key] = (
                    image_patch, prepared_patch, prepared_torch, wepl_patch
                )
        for offset in range(0, len(conditions), condition_batch_size):
            batch_conditions = conditions[offset : offset + condition_batch_size]
            if roi_activity is not None:
                candidate_local = [
                    index
                    for index in range(len(batch_conditions))
                    if roi_activity[start_index, offset + index]
                ]
                if not candidate_local:
                    continue
            else:
                candidate_local = list(range(len(batch_conditions)))
            build_conditions = [batch_conditions[index] for index in candidate_local]
            if prepared_torch is not None:
                batch_inputs_torch = _build_proton_channels_torch(
                    prepared_torch, build_conditions
                )
                if skip_empty_ray:
                    active_tensor = torch.nonzero(
                        batch_inputs_torch[:, 2].amax(dim=(1, 2, 3)) >= 1.0e-6
                    ).flatten()
                else:
                    active_tensor = torch.arange(len(build_conditions), device=device)
                if active_tensor.numel() == 0:
                    continue
                active_indices = [
                    candidate_local[index] for index in active_tensor.cpu().tolist()
                ]
                inputs = batch_inputs_torch.index_select(0, active_tensor)
            else:
                batch_inputs = build_proton_channels_batch(
                    image_patch,
                    start,
                    geometry,
                    build_conditions,
                    modality=modality,
                    intensity_bounds=bounds,
                    prepared=prepared_patch,
                    wepl_patch=wepl_patch,
                )
                kept_local = [
                    index for index, channels in enumerate(batch_inputs)
                    if not skip_empty_ray or float(channels[2].max()) >= 1.0e-6
                ]
                if not kept_local:
                    continue
                active_indices = [candidate_local[index] for index in kept_local]
                inputs = torch.from_numpy(batch_inputs[kept_local]).to(
                    device, non_blocking=True
                )
            pending_inputs.append(inputs)
            pending_rows.extend(
                (offset + index, start) for index in active_indices
            )
            if sum(item.shape[0] for item in pending_inputs) >= condition_batch_size:
                flush_pending()
    flush_pending()

    predictions_array = (
        outputs / torch.clamp_min(total_weight, 1.0e-8).unsqueeze(0)
    ).cpu().numpy()
    if mask_outside_body:
        threshold = -1000.0 if modality == "ct" else bounds[0]
        predictions_array[:, image <= threshold] = 0.0
    if relative_cutoff > 0.0:
        for prediction in predictions_array:
            peak = float(prediction.max())
            if peak > 0.0:
                prediction[prediction < relative_cutoff * peak] = 0.0
    return [np.asarray(prediction, dtype=np.float32) for prediction in predictions_array]
