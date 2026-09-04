from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from .conditioning import (
    PhotonCondition,
    ConditionChannelContext,
    SpatialGeometry,
    build_condition_channels,
    build_condition_channels_batch,
    build_condition_channel_context,
    build_condition_channels_from_context,
    mri_foreground_bounds,
)
from .dataset import ManifestRecord, crop_with_padding, lookup_condition
from .dataset_index import read_mha_header
from .mha import load_mha_array


@dataclass
class PreparedConditionedInference:
    """Reusable sliding-window state for all control points of one MRI."""

    image_shape_zyx: tuple[int, int, int]
    starts: list[tuple[int, int, int]]
    channel_contexts: list[ConditionChannelContext]
    patch_slices: list[tuple[tuple[slice, ...], tuple[slice, ...]]]
    patch_weight: torch.Tensor
    total_weight: torch.Tensor


@torch.inference_mode()
def prepare_conditioned_inference(
    image: NDArray[np.floating],
    geometry: SpatialGeometry,
    *,
    device: torch.device,
    patch_size_zyx: tuple[int, int, int],
    intensity_bounds: tuple[float, float] | None = None,
    overlap: float = 0.5,
) -> PreparedConditionedInference:
    """Prepare MRI-only state once and reuse it across control-point chunks."""
    image = np.asarray(image, dtype=np.float32)
    intensity_bounds = intensity_bounds or mri_foreground_bounds(image)
    starts = [
        (z, y, x)
        for z in sliding_window_starts(image.shape[0], patch_size_zyx[0], overlap)
        for y in sliding_window_starts(image.shape[1], patch_size_zyx[1], overlap)
        for x in sliding_window_starts(image.shape[2], patch_size_zyx[2], overlap)
    ]
    channel_contexts = []
    patch_slices = []
    for start in starts:
        image_patch = crop_with_padding(image, start, patch_size_zyx, pad_value=0.0)
        channel_contexts.append(
            build_condition_channel_context(
                image_patch,
                start,
                geometry,
                intensity_bounds=intensity_bounds,
            )
        )
        patch_slices.append(_patch_slices(image.shape, patch_size_zyx, start))

    patch_weight = torch.from_numpy(gaussian_blend_weight(patch_size_zyx)).to(device)
    total_weight = torch.zeros(image.shape, dtype=torch.float32, device=device)
    for destination_slices, source_slices in patch_slices:
        total_weight[destination_slices] += patch_weight[source_slices]
    return PreparedConditionedInference(
        image_shape_zyx=tuple(int(value) for value in image.shape),
        starts=starts,
        channel_contexts=channel_contexts,
        patch_slices=patch_slices,
        patch_weight=patch_weight,
        total_weight=total_weight,
    )


@torch.inference_mode()
def warmup_model(
    model: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    in_channels: int,
    patch_size_zyx: tuple[int, int, int],
    amp: bool = True,
) -> None:
    """Initialize CUDA/cuDNN kernels before the timed invoke phase."""
    if device.type != "cuda":
        return
    inputs = torch.zeros(
        (batch_size, in_channels, *patch_size_zyx),
        # Real inputs arrive as FP32 and autocast selects FP16 kernels inside
        # the model. Matching the external dtype also prevents torch.compile
        # from creating a second graph during the timed invoke phase.
        dtype=torch.float32,
        device=device,
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        model(inputs)
    torch.cuda.synchronize(device)
    del inputs


def sliding_window_starts(size: int, patch: int, overlap: float) -> list[int]:
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if size <= patch:
        return [-((patch - size) // 2)]
    stride = max(1, int(round(patch * (1.0 - overlap))))
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def gaussian_blend_weight(patch_size_zyx: Sequence[int]) -> NDArray[np.float32]:
    axes = []
    for size in patch_size_zyx:
        coordinate = np.linspace(-1.0, 1.0, int(size), dtype=np.float32)
        axes.append(np.exp(-0.5 * (coordinate / 0.5) ** 2))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return np.maximum(weight, 1.0e-3).astype(np.float32)


def _accumulate_patch(
    output: NDArray[np.float32],
    total_weight: NDArray[np.float32],
    prediction: NDArray[np.float32],
    patch_weight: NDArray[np.float32],
    start_zyx: Sequence[int],
) -> None:
    start = np.asarray(start_zyx, dtype=np.int64)
    end = start + np.asarray(prediction.shape, dtype=np.int64)
    destination_start = np.maximum(start, 0)
    destination_end = np.minimum(end, np.asarray(output.shape))
    source_start = destination_start - start
    source_end = source_start + (destination_end - destination_start)
    destination_slices = tuple(
        slice(int(begin), int(finish))
        for begin, finish in zip(destination_start, destination_end)
    )
    source_slices = tuple(
        slice(int(begin), int(finish)) for begin, finish in zip(source_start, source_end)
    )
    output[destination_slices] += prediction[source_slices] * patch_weight[source_slices]
    total_weight[destination_slices] += patch_weight[source_slices]


def _accumulate_prediction_only(
    output: NDArray[np.float32],
    prediction: NDArray[np.float32],
    patch_weight: NDArray[np.float32],
    start_zyx: Sequence[int],
) -> None:
    start = np.asarray(start_zyx, dtype=np.int64)
    end = start + np.asarray(prediction.shape, dtype=np.int64)
    destination_start = np.maximum(start, 0)
    destination_end = np.minimum(end, np.asarray(output.shape))
    source_start = destination_start - start
    source_end = source_start + (destination_end - destination_start)
    destination_slices = tuple(
        slice(int(begin), int(finish))
        for begin, finish in zip(destination_start, destination_end)
    )
    source_slices = tuple(
        slice(int(begin), int(finish)) for begin, finish in zip(source_start, source_end)
    )
    output[destination_slices] += prediction[source_slices] * patch_weight[source_slices]


def _patch_slices(
    volume_shape_zyx: Sequence[int],
    patch_shape_zyx: Sequence[int],
    start_zyx: Sequence[int],
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    start = np.asarray(start_zyx, dtype=np.int64)
    end = start + np.asarray(patch_shape_zyx, dtype=np.int64)
    destination_start = np.maximum(start, 0)
    destination_end = np.minimum(end, np.asarray(volume_shape_zyx))
    source_start = destination_start - start
    source_end = source_start + (destination_end - destination_start)
    destination_slices = tuple(
        slice(int(begin), int(finish))
        for begin, finish in zip(destination_start, destination_end)
    )
    source_slices = tuple(
        slice(int(begin), int(finish)) for begin, finish in zip(source_start, source_end)
    )
    return destination_slices, source_slices


@torch.inference_mode()
def predict_record_volume(
    model: torch.nn.Module,
    record: ManifestRecord,
    *,
    device: torch.device,
    patch_size_zyx: tuple[int, int, int],
    dose_scale: float,
    intensity_bounds: tuple[float, float] | None = None,
    overlap: float = 0.5,
    batch_size: int = 1,
    amp: bool = True,
    skip_empty_aperture: bool = False,
    mask_outside_body: bool = False,
    pad_to_batch_size: bool = False,
    include_physics_priors: bool = False,
) -> NDArray[np.float32]:
    condition: PhotonCondition = lookup_condition(record)
    return predict_conditioned_volume(
        model,
        image_path=record.image_path,
        condition=condition,
        device=device,
        patch_size_zyx=patch_size_zyx,
        dose_scale=dose_scale,
        intensity_bounds=intensity_bounds,
        overlap=overlap,
        batch_size=batch_size,
        amp=amp,
        skip_empty_aperture=skip_empty_aperture,
        mask_outside_body=mask_outside_body,
        pad_to_batch_size=pad_to_batch_size,
        include_physics_priors=include_physics_priors,
    )


@torch.inference_mode()
def predict_conditioned_volume(
    model: torch.nn.Module,
    *,
    image_path: Path,
    condition: PhotonCondition,
    device: torch.device,
    patch_size_zyx: tuple[int, int, int],
    dose_scale: float,
    intensity_bounds: tuple[float, float] | None = None,
    overlap: float = 0.5,
    batch_size: int = 1,
    amp: bool = True,
    skip_empty_aperture: bool = False,
    mask_outside_body: bool = False,
    pad_to_batch_size: bool = False,
    include_physics_priors: bool = False,
) -> NDArray[np.float32]:
    image = np.asarray(load_mha_array(image_path), dtype=np.float32)
    geometry = SpatialGeometry.from_mha_header(read_mha_header(image_path))
    return predict_conditioned_array(
        model,
        image=image,
        geometry=geometry,
        condition=condition,
        device=device,
        patch_size_zyx=patch_size_zyx,
        dose_scale=dose_scale,
        intensity_bounds=intensity_bounds,
        overlap=overlap,
        batch_size=batch_size,
        amp=amp,
        skip_empty_aperture=skip_empty_aperture,
        mask_outside_body=mask_outside_body,
        pad_to_batch_size=pad_to_batch_size,
        include_physics_priors=include_physics_priors,
    )


@torch.inference_mode()
def predict_conditioned_array(
    model: torch.nn.Module,
    *,
    image: NDArray[np.floating],
    geometry: SpatialGeometry,
    condition: PhotonCondition,
    device: torch.device,
    patch_size_zyx: tuple[int, int, int],
    dose_scale: float,
    intensity_bounds: tuple[float, float] | None = None,
    overlap: float = 0.5,
    batch_size: int = 1,
    amp: bool = True,
    skip_empty_aperture: bool = False,
    mask_outside_body: bool = False,
    pad_to_batch_size: bool = False,
    include_physics_priors: bool = False,
) -> NDArray[np.float32]:
    image = np.asarray(image, dtype=np.float32)
    intensity_bounds = intensity_bounds or mri_foreground_bounds(image)
    starts = [
        (z, y, x)
        for z in sliding_window_starts(image.shape[0], patch_size_zyx[0], overlap)
        for y in sliding_window_starts(image.shape[1], patch_size_zyx[1], overlap)
        for x in sliding_window_starts(image.shape[2], patch_size_zyx[2], overlap)
    ]
    output = np.zeros(image.shape, dtype=np.float32)
    total_weight = np.zeros(image.shape, dtype=np.float32)
    patch_weight = gaussian_blend_weight(patch_size_zyx)

    model.eval()
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        batch_inputs = []
        for start in batch_starts:
            image_patch = crop_with_padding(image, start, patch_size_zyx, pad_value=0.0)
            batch_inputs.append(
                build_condition_channels(
                    image_patch,
                    patch_start_zyx=start,
                    geometry=geometry,
                    condition=condition,
                    intensity_bounds=intensity_bounds,
                    include_physics_priors=include_physics_priors,
                )
            )
        active_indices = [
            index
            for index, channels in enumerate(batch_inputs)
            if not skip_empty_aperture or bool(np.any(channels[2] > 0.0))
        ]
        if not active_indices:
            continue
        active_inputs = np.stack([batch_inputs[index] for index in active_indices])
        active_count = len(active_indices)
        if pad_to_batch_size and active_count < batch_size:
            padding = np.zeros(
                (batch_size - active_count, *active_inputs.shape[1:]),
                dtype=active_inputs.dtype,
            )
            active_inputs = np.concatenate((active_inputs, padding), axis=0)
        tensor = torch.from_numpy(active_inputs).to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=amp and device.type in {"cuda", "cpu"},
        ):
            batch_predictions = model(tensor)
        predictions = (
            batch_predictions[:active_count, 0].float().cpu().numpy() * dose_scale
        )
        for prediction, active_index in zip(predictions, active_indices):
            start = batch_starts[active_index]
            _accumulate_patch(output, total_weight, prediction, patch_weight, start)
    prediction = output / np.maximum(total_weight, 1.0e-8)
    if mask_outside_body:
        prediction[image <= 0.0] = 0.0
    return prediction


@torch.inference_mode()
def predict_conditioned_arrays(
    model: torch.nn.Module,
    *,
    image: NDArray[np.floating],
    geometry: SpatialGeometry,
    conditions: Sequence[PhotonCondition],
    device: torch.device,
    patch_size_zyx: tuple[int, int, int],
    dose_scale: float,
    intensity_bounds: tuple[float, float] | None = None,
    overlap: float = 0.5,
    condition_batch_size: int = 4,
    amp: bool = True,
    skip_empty_aperture: bool = False,
    mask_outside_body: bool = False,
    prepared: PreparedConditionedInference | None = None,
    pad_to_batch_size: bool = False,
    include_physics_priors: bool = False,
) -> list[NDArray[np.float32]]:
    """Predict several control points for one MRI in batched GPU forwards.

    Keep ``conditions`` reasonably small (for example 4--8) because every
    output dose volume is accumulated in host memory until the chunk ends.
    """
    if not conditions:
        return []
    if condition_batch_size < 1:
        raise ValueError("condition_batch_size must be positive")

    image = np.asarray(image, dtype=np.float32)
    if prepared is None:
        prepared = prepare_conditioned_inference(
            image,
            geometry,
            device=device,
            patch_size_zyx=patch_size_zyx,
            intensity_bounds=intensity_bounds,
            overlap=overlap,
        )
    if prepared.image_shape_zyx != tuple(image.shape):
        raise ValueError("prepared inference shape does not match MRI shape")
    outputs = torch.zeros(
        (len(conditions), *image.shape), dtype=torch.float32, device=device
    )

    model.eval()
    for channel_context, slices in zip(
        prepared.channel_contexts, prepared.patch_slices
    ):
        destination_slices, source_slices = slices
        for offset in range(0, len(conditions), condition_batch_size):
            batch_conditions = conditions[offset : offset + condition_batch_size]
            batch_inputs = build_condition_channels_from_context(
                channel_context,
                batch_conditions,
                include_physics_priors=include_physics_priors,
            )
            active_indices = [
                index
                for index, channels in enumerate(batch_inputs)
                if not skip_empty_aperture or bool(np.any(channels[2] > 0.0))
            ]
            if not active_indices:
                continue
            active_inputs = batch_inputs[active_indices]
            active_count = len(active_indices)
            if pad_to_batch_size and active_count < condition_batch_size:
                padding = np.zeros(
                    (condition_batch_size - active_count, *active_inputs.shape[1:]),
                    dtype=active_inputs.dtype,
                )
                active_inputs = np.concatenate((active_inputs, padding), axis=0)
            tensor = torch.from_numpy(active_inputs).to(
                device, non_blocking=True
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp and device.type in {"cuda", "cpu"},
            ):
                batch_predictions = model(tensor)
            predictions = batch_predictions[:active_count, 0].float() * dose_scale
            weighted_predictions = (
                predictions[(slice(None), *source_slices)]
                * prepared.patch_weight[source_slices].unsqueeze(0)
            )
            output_chunk = outputs[
                (slice(offset, offset + len(batch_conditions)), *destination_slices)
            ]
            if len(active_indices) == len(batch_conditions):
                output_chunk += weighted_predictions
            else:
                output_chunk[active_indices] += weighted_predictions

    predictions_array = (
        outputs / torch.clamp_min(prepared.total_weight, 1.0e-8).unsqueeze(0)
    ).cpu().numpy()
    predictions = [prediction for prediction in predictions_array]
    if mask_outside_body:
        outside_body = image <= 0.0
        for prediction in predictions:
            prediction[outside_body] = 0.0
    return predictions
