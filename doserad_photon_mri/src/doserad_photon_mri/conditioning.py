from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


HU_CALIBRATION = np.asarray(
    [-1024.0, -999.0, -200.0, -199.0, -10.0, -9.0, 120.0, 121.0, 3000.0, 4000.0],
    dtype=np.float32,
)
DENSITY_CALIBRATION = np.asarray(
    [0.0012, 0.00121, 0.8043754, 0.8183035, 1.006579, 0.9966749,
     1.126553, 1.095097, 3.027294, 3.698428],
    dtype=np.float32,
)


def hu_to_mass_density(ct: NDArray[np.floating]) -> NDArray[np.float32]:
    """Official DoseRAD2026 CT-density mapping used as MRI training supervision."""
    return np.asarray(
        np.interp(
            np.asarray(ct, dtype=np.float32), HU_CALIBRATION, DENSITY_CALIBRATION
        ),
        dtype=np.float32,
    )


def mri_foreground_bounds(
    image: NDArray[np.floating], lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[float, float]:
    """Robust per-volume MRI scaling bounds, excluding zero background."""
    values = np.asarray(image, dtype=np.float32)
    foreground = values[values > 0.0]
    if foreground.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(foreground, [lower_percentile, upper_percentile])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(foreground.min())
        high = float(foreground.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


@dataclass(frozen=True)
class SpatialGeometry:
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: tuple[float, ...]

    @classmethod
    def from_mha_header(cls, header: dict[str, str]) -> "SpatialGeometry":
        spacing = tuple(float(value) for value in header["ElementSpacing"].split())
        origin_text = header.get("Offset", header.get("Position", "0 0 0"))
        origin = tuple(float(value) for value in origin_text.split())
        direction_text = header.get("TransformMatrix", "1 0 0 0 1 0 0 0 1")
        direction = tuple(float(value) for value in direction_text.split())
        if len(spacing) != 3 or len(origin) != 3 or len(direction) != 9:
            raise ValueError("only 3D MHA geometry is supported")
        return cls(spacing_xyz=spacing, origin_xyz=origin, direction=direction)


@dataclass(frozen=True)
class ConditionChannelContext:
    """Patch data shared by every beam control point for one MRI."""

    image_normalized: NDArray[np.float32]
    body: NDArray[np.float32]
    physical_x: NDArray[np.float32]
    physical_y: NDArray[np.float32]
    physical_z: NDArray[np.float32]


@dataclass(frozen=True)
class PhotonCondition:
    gantry_angle_deg: float
    iso_center_xyz: tuple[float, float, float]
    mlc_left_int_mm: tuple[float, ...]
    mlc_right_int_mm: tuple[float, ...]
    sad_mm: float = 1000.0
    mlc_leaf_width_mm: float = 5.0

    @classmethod
    def from_json(cls, beam: dict, control_point: dict) -> "PhotonCondition":
        leaf_pairs = int(beam["num_mlc_leaf_pairs"])
        left = tuple(float(value) for value in control_point["mlc_left_int_mm"])
        right = tuple(float(value) for value in control_point["mlc_right_int_mm"])
        if len(left) != leaf_pairs or len(right) != leaf_pairs:
            raise ValueError("MLC array length does not match num_mlc_leaf_pairs")
        return cls(
            gantry_angle_deg=float(control_point["gantry_angle"]),
            iso_center_xyz=tuple(float(value) for value in beam["iso_center"]),
            mlc_left_int_mm=left,
            mlc_right_int_mm=right,
            sad_mm=float(beam.get("SAD", 1000.0)),
        )

    @property
    def leaf_pairs(self) -> int:
        return len(self.mlc_left_int_mm)

    def transformed_leaf_edges(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Match the MLC convention used by the official pyRadPlan baseline."""
        raw_left = np.asarray(self.mlc_left_int_mm, dtype=np.float32)
        raw_right = np.asarray(self.mlc_right_int_mm, dtype=np.float32)
        left = -np.flip(raw_right) + 0.5
        right = -np.flip(raw_left) + 0.5
        return left, right


def _physical_coordinates(
    patch_start_zyx: Sequence[int],
    patch_shape_zyx: Sequence[int],
    geometry: SpatialGeometry,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    start_z, start_y, start_x = (int(value) for value in patch_start_zyx)
    size_z, size_y, size_x = (int(value) for value in patch_shape_zyx)
    sx, sy, sz = geometry.spacing_xyz
    ox, oy, oz = geometry.origin_xyz

    index_x = np.arange(start_x, start_x + size_x, dtype=np.float32)
    index_y = np.arange(start_y, start_y + size_y, dtype=np.float32)
    index_z = np.arange(start_z, start_z + size_z, dtype=np.float32)
    zz, yy, xx = np.meshgrid(index_z, index_y, index_x, indexing="ij", sparse=True)

    direction = np.asarray(geometry.direction, dtype=np.float32).reshape(3, 3)
    # Physical point = origin + direction @ (index * spacing). Dataset images
    # currently have identity direction, but retaining this keeps the loader safe.
    vx = xx * sx
    vy = yy * sy
    vz = zz * sz
    physical_x = ox + direction[0, 0] * vx + direction[0, 1] * vy + direction[0, 2] * vz
    physical_y = oy + direction[1, 0] * vx + direction[1, 1] * vy + direction[1, 2] * vz
    physical_z = oz + direction[2, 0] * vx + direction[2, 1] * vy + direction[2, 2] * vz
    return physical_x, physical_y, physical_z


def build_condition_channels(
    image_patch: NDArray[np.floating],
    patch_start_zyx: Sequence[int],
    geometry: SpatialGeometry,
    condition: PhotonCondition,
    *,
    intensity_bounds: tuple[float, float] | None = None,
    coordinate_scale_mm: float = 400.0,
    include_physics_priors: bool = False,
) -> NDArray[np.float32]:
    """Build MRI, body, aperture, depth, lateral and SI channels."""
    return build_condition_channels_batch(
        image_patch,
        patch_start_zyx,
        geometry,
        [condition],
        intensity_bounds=intensity_bounds,
        coordinate_scale_mm=coordinate_scale_mm,
        include_physics_priors=include_physics_priors,
    )[0]


def build_condition_channels_batch(
    image_patch: NDArray[np.floating],
    patch_start_zyx: Sequence[int],
    geometry: SpatialGeometry,
    conditions: Sequence[PhotonCondition],
    *,
    intensity_bounds: tuple[float, float] | None = None,
    coordinate_scale_mm: float = 400.0,
    include_physics_priors: bool = False,
) -> NDArray[np.float32]:
    """Build condition tensors while reusing MRI and physical coordinates.

    The returned shape is ``(conditions, 6, z, y, x)``. Grouping control
    points this way removes repeated MRI preprocessing and enables one GPU
    forward pass for several dose maps of the same patient.
    """
    context = build_condition_channel_context(
        image_patch,
        patch_start_zyx,
        geometry,
        intensity_bounds=intensity_bounds,
    )
    return build_condition_channels_from_context(
        context,
        conditions,
        coordinate_scale_mm=coordinate_scale_mm,
        include_physics_priors=include_physics_priors,
    )


def build_condition_channel_context(
    image_patch: NDArray[np.floating],
    patch_start_zyx: Sequence[int],
    geometry: SpatialGeometry,
    *,
    intensity_bounds: tuple[float, float] | None = None,
) -> ConditionChannelContext:
    """Prepare MRI normalization and coordinates once for a sliding patch."""
    if image_patch.ndim != 3:
        raise ValueError(f"expected 3D MRI patch, got shape {image_patch.shape}")
    image = np.asarray(image_patch, dtype=np.float32)
    low, high = intensity_bounds or mri_foreground_bounds(image)
    if high <= low:
        raise ValueError("invalid MRI normalization bounds")
    image_normalized = np.clip(image, low, high)
    image_normalized = 2.0 * (image_normalized - low) / (high - low) - 1.0
    body = (image > 0.0).astype(np.float32)
    image_normalized[body == 0.0] = -1.0

    px, py, pz = _physical_coordinates(patch_start_zyx, image.shape, geometry)
    return ConditionChannelContext(
        image_normalized=image_normalized,
        body=body,
        physical_x=px,
        physical_y=py,
        physical_z=pz,
    )


def build_condition_channels_from_context(
    context: ConditionChannelContext,
    conditions: Sequence[PhotonCondition],
    *,
    coordinate_scale_mm: float = 400.0,
    include_physics_priors: bool = False,
) -> NDArray[np.float32]:
    """Build condition-specific channels from a reusable patch context."""
    if not conditions:
        raise ValueError("conditions must not be empty")
    output_shape = context.image_normalized.shape
    channel_count = 10 if include_physics_priors else 6
    channels = np.empty(
        (len(conditions), channel_count, *output_shape), dtype=np.float32
    )
    channels[:, 0] = context.image_normalized
    channels[:, 1] = context.body

    for condition_index, condition in enumerate(conditions):
        iso_x, iso_y, iso_z = condition.iso_center_xyz
        rel_x = context.physical_x - iso_x
        rel_y = context.physical_y - iso_y
        rel_z = context.physical_z - iso_z

        gantry = math.radians(condition.gantry_angle_deg)
        direction_x = -math.sin(gantry)
        direction_y = math.cos(gantry)
        lateral_x = -math.cos(gantry)
        lateral_y = -math.sin(gantry)

        depth = rel_x * direction_x + rel_y * direction_y
        lateral = rel_x * lateral_x + rel_y * lateral_y
        superior_inferior = rel_z

        left_edges, right_edges = condition.transformed_leaf_edges()
        source_to_voxel = condition.sad_mm + depth
        valid_projection = source_to_voxel > 1.0e-3
        projection_scale = condition.sad_mm / np.maximum(source_to_voxel, 1.0e-3)
        projected_lateral = lateral * projection_scale
        projected_si = superior_inferior * projection_scale
        field_half_height = condition.leaf_pairs * condition.mlc_leaf_width_mm / 2.0
        leaf_indices = np.floor(
            (projected_si + field_half_height) / condition.mlc_leaf_width_mm
        ).astype(np.int32)
        valid_leaf = (
            valid_projection
            & (leaf_indices >= 0)
            & (leaf_indices < condition.leaf_pairs)
        )
        safe_indices = np.clip(leaf_indices, 0, condition.leaf_pairs - 1)
        aperture = (
            valid_leaf
            & (projected_lateral >= left_edges[safe_indices])
            & (projected_lateral <= right_edges[safe_indices])
        )

        channels[condition_index, 2] = np.broadcast_to(aperture, output_shape)
        channels[condition_index, 3] = np.broadcast_to(
            np.clip(depth / coordinate_scale_mm, -1.0, 1.0), output_shape
        )
        channels[condition_index, 4] = np.broadcast_to(
            np.clip(lateral / coordinate_scale_mm, -1.0, 1.0), output_shape
        )
        channels[condition_index, 5] = np.broadcast_to(
            np.clip(superior_inferior / coordinate_scale_mm, -1.0, 1.0),
            output_shape,
        )
        if include_physics_priors:
            left_margin = projected_lateral - left_edges[safe_indices]
            right_margin = right_edges[safe_indices] - projected_lateral
            vertical_margin = field_half_height - np.abs(projected_si)
            signed_edge_mm = np.minimum(
                np.minimum(left_margin, right_margin), vertical_margin
            )
            signed_edge_mm = np.where(valid_projection, signed_edge_mm, -40.0)
            inverse_square = np.clip(
                (condition.sad_mm / np.maximum(source_to_voxel, 1.0e-3)) ** 2,
                0.0,
                4.0,
            )
            opening_width = np.maximum(
                right_edges[safe_indices] - left_edges[safe_indices], 0.0
            )
            opening_width = np.where(valid_leaf, opening_width, 0.0)
            soft_aperture = 1.0 / (
                1.0 + np.exp(-np.clip(signed_edge_mm / 3.0, -20.0, 20.0))
            )
            channels[condition_index, 6] = np.broadcast_to(
                np.clip(signed_edge_mm / 40.0, -1.0, 1.0), output_shape
            )
            channels[condition_index, 7] = np.broadcast_to(
                inverse_square, output_shape
            )
            channels[condition_index, 8] = np.broadcast_to(
                np.clip(opening_width / 200.0, 0.0, 1.0), output_shape
            )
            channels[condition_index, 9] = np.broadcast_to(
                soft_aperture * inverse_square * context.body, output_shape
            )
    return channels
