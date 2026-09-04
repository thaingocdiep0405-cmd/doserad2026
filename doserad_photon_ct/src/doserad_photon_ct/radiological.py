"""Radiological (water-equivalent) depth along the photon beam.

The gantry rotates in the transverse plane, so every z-slice is an
independent 2D problem: rotate the mass-density slice so the beam runs
along one grid axis, take a cumulative sum times the step length, and
rotate the result back. Rays are treated as parallel; beam divergence at
SAD 1000 mm changes the path direction by well under 12 degrees inside
the field and is left for the network to absorb.

Everything is computed with torch so the same code runs on CPU inside
dataloader workers and on GPU during full-volume inference.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray


def compute_radiological_depth(
    density_zyx: NDArray[np.floating],
    spacing_xyz: tuple[float, float, float],
    gantry_angle_deg: float,
    *,
    downsample_xy: int = 2,
    device: torch.device | str | None = None,
) -> NDArray[np.float32]:
    """Water-equivalent depth in mm for every voxel of ``density_zyx``.

    ``density_zyx`` may be a z-slab of the full volume: slices are
    independent, but it must always cover the full x/y extent because
    rays traverse the whole transverse plane.
    """
    sx, sy, _ = (float(value) for value in spacing_xyz)
    if abs(sx - sy) > 1.0e-3 * max(sx, sy):
        raise ValueError(f"anisotropic in-plane spacing is unsupported: {sx} != {sy}")
    if downsample_xy < 1:
        raise ValueError("downsample_xy must be >= 1")

    with torch.no_grad():
        density = torch.as_tensor(
            np.ascontiguousarray(density_zyx), dtype=torch.float32, device=device
        )
        if density.ndim != 3:
            raise ValueError(f"expected (z, y, x) density, got shape {tuple(density.shape)}")
        slices, full_y, full_x = density.shape
        density = density.unsqueeze(1)  # (z, 1, y, x)

        if downsample_xy > 1:
            density = F.avg_pool2d(
                density, downsample_xy, ceil_mode=True, count_include_pad=False
            )
        step_mm = sx * downsample_xy
        height, width = density.shape[-2:]

        # Rotated canvas large enough for the slice diagonal. An odd size,
        # integer embedding margins and rotation about the integer centre
        # keep every sampling position on the pixel grid for gantry angles
        # that are multiples of 90 degrees, so those common angles suffer
        # no bilinear blur (blur would accumulate through the cumsum).
        canvas = (int(math.ceil(math.hypot(height, width))) + 2) | 1
        margin_y = (canvas - height) // 2
        margin_x = (canvas - width) // 2
        center = (canvas - 1) // 2
        gantry = math.radians(float(gantry_angle_deg))
        # Beam direction in (y, x) index space equals the physical
        # (cos g, -sin g) because the grid direction is identity with
        # positive spacing.
        beam_y, beam_x = math.cos(gantry), -math.sin(gantry)

        axis = torch.arange(canvas, dtype=torch.float32, device=density.device) - center
        uu, vv = torch.meshgrid(axis, axis, indexing="ij")
        # Rotation with +u aligned to the beam: (u, v) -> (y, x).
        sample_y = beam_y * uu - beam_x * vv + center - margin_y
        sample_x = beam_x * uu + beam_y * vv + center - margin_x
        forward_grid = torch.stack(
            (
                (sample_x + 0.5) / width * 2.0 - 1.0,
                (sample_y + 0.5) / height * 2.0 - 1.0,
            ),
            dim=-1,
        ).unsqueeze(0).expand(slices, -1, -1, -1)

        rotated = F.grid_sample(
            density,
            forward_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        # Midpoint rule: integrate up to each voxel centre, not its exit face.
        depth_rotated = (torch.cumsum(rotated, dim=2) - 0.5 * rotated) * step_mm

        axis_y = (
            torch.arange(height, dtype=torch.float32, device=density.device)
            + margin_y
            - center
        )
        axis_x = (
            torch.arange(width, dtype=torch.float32, device=density.device)
            + margin_x
            - center
        )
        dy, dx = torch.meshgrid(axis_y, axis_x, indexing="ij")
        sample_u = beam_y * dy + beam_x * dx + center
        sample_v = -beam_x * dy + beam_y * dx + center
        inverse_grid = torch.stack(
            (
                (sample_v + 0.5) / canvas * 2.0 - 1.0,
                (sample_u + 0.5) / canvas * 2.0 - 1.0,
            ),
            dim=-1,
        ).unsqueeze(0).expand(slices, -1, -1, -1)

        depth_half = F.grid_sample(
            depth_rotated,
            inverse_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        if downsample_xy > 1 or depth_half.shape[-2:] != (full_y, full_x):
            depth_half = F.interpolate(
                depth_half, size=(full_y, full_x), mode="bilinear", align_corners=False
            )
        return depth_half.squeeze(1).clamp_(min=0.0).cpu().numpy()
