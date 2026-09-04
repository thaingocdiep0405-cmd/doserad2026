from __future__ import annotations

import math

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray


def masked_beam_mae(
    prediction: NDArray[np.floating], target: NDArray[np.floating]
) -> float:
    target_max = float(np.max(target))
    if target_max <= 0:
        return float("nan")
    mask = target >= 0.1 * target_max
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(prediction[mask] - target[mask])) / target_max)


def beam_direction(gantry_angle_deg: float) -> NDArray[np.float64]:
    gantry = math.radians(float(gantry_angle_deg))
    return np.asarray([-math.sin(gantry), math.cos(gantry), 0.0], dtype=np.float64)


def compute_idd_curve(
    dose_zyx: NDArray[np.floating],
    direction_xyz: NDArray[np.floating],
    spacing_xyz: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Match the official transverse-plane IDD resampling using SciPy."""
    direction = np.asarray(direction_xyz, dtype=np.float64)
    if abs(float(direction[2])) > 1.0e-9:
        raise ValueError("beam leaves the transverse plane; z cannot be summed")

    plane_yx = np.asarray(dose_zyx).sum(axis=0, dtype=np.float64)
    ny, nx = plane_yx.shape
    sx, sy = float(spacing_xyz[0]), float(spacing_xyz[1])
    center_x = (nx - 1) * sx / 2.0
    center_y = (ny - 1) * sy / 2.0
    step = max(sx, sy)
    size = int(math.ceil(math.hypot(nx * sx, ny * sy) / step)) + 1
    origin = -(size - 1) * step / 2.0

    source = sitk.GetImageFromArray(plane_yx)
    source.SetSpacing((sx, sy))
    transform = sitk.Euler2DTransform()
    transform.SetCenter((0.0, 0.0))
    transform.SetAngle(math.atan2(float(direction[1]), float(direction[0])))
    transform.SetTranslation((center_x, center_y))
    aligned = sitk.Resample(
        source,
        (size, size),
        transform,
        sitk.sitkLinear,
        (origin, origin),
        (step, step),
        (1.0, 0.0, 0.0, 1.0),
        0.0,
        sitk.sitkFloat64,
    )
    return sitk.GetArrayFromImage(aligned).sum(axis=0)


def idd_curve_distance(
    prediction: NDArray[np.floating],
    target: NDArray[np.floating],
    direction_xyz: NDArray[np.floating],
    spacing_xyz: tuple[float, float, float],
) -> float:
    predicted_curve = compute_idd_curve(prediction, direction_xyz, spacing_xyz)
    target_curve = compute_idd_curve(target, direction_xyz, spacing_xyz)
    target_max = float(np.max(target_curve))
    if target_max <= 0:
        return float("nan")
    return float(
        np.sqrt(np.mean((predicted_curve / target_max - target_curve / target_max) ** 2))
    )
