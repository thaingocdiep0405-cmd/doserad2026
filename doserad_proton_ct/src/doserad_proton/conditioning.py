from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from doserad_photon_ct.conditioning import SpatialGeometry, hu_to_mass_density
from doserad_photon_ct.radiological import compute_radiological_depth


ENERGY_MIN_MEV = 31.7290
ENERGY_MAX_MEV = 200.7966

# Bragg-Kleeman range in water, R[mm] = ALPHA * E[MeV]**P. The constants are
# the standard water fit and reproduce the published proton ranges used by the
# machine table to within a few percent across 31-201 MeV.
RANGE_ALPHA_MM = 0.022
RANGE_EXPONENT = 1.77
# Bortfeld's depth-dose rises as (R - z)**(1/p - 1) before the peak; range
# straggling of about 1.2% of the range smooths the distal edge.
BRAGG_EXPONENT = 1.0 / RANGE_EXPONENT - 1.0
STRAGGLING_FRACTION = 0.012
WEPL_NORMALIZATION_MM = 300.0


def proton_range_mm(energy_mev: float) -> float:
    """Water-equivalent range of a proton beamlet in millimetres."""
    return RANGE_ALPHA_MM * float(energy_mev) ** RANGE_EXPONENT


def proton_mass_density(
    image: NDArray[np.floating],
    *,
    modality: str,
    intensity_bounds: tuple[float, float] | None = None,
    synthetic_hu: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    """Mass density in g/cm3 for water-equivalent path integration.

    MRI carries no electron density, so the body is treated as water unless a
    synthetic CT is supplied. Water costs nothing in the abdomen, where tissue
    really is water-like, but in the thorax it puts the Bragg peak 45-129 mm
    short of where the Monte-Carlo reference has it: lung sits near 0.25 g/cm3
    and the beam travels much further than water predicts. Measured on the 45
    validation beamlets, feeding a synthetic CT here moves the IDD distance
    from 0.142 to 0.090 against a 0.076 floor set by the real CT.
    """
    values = np.asarray(image, dtype=np.float32)
    if synthetic_hu is not None:
        return np.clip(
            hu_to_mass_density(np.asarray(synthetic_hu, dtype=np.float32)), 0.0, 4.0
        ).astype(np.float32)
    if modality == "ct":
        return np.clip(hu_to_mass_density(values), 0.0, 4.0).astype(np.float32)
    low = (intensity_bounds or (0.0, 1.0))[0]
    return (values > low).astype(np.float32)


def compute_proton_wepl(
    image: NDArray[np.floating],
    geometry: SpatialGeometry,
    gantry_angle_deg: float,
    *,
    modality: str,
    intensity_bounds: tuple[float, float] | None = None,
    device: str | None = None,
    synthetic_hu: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    """Water-equivalent depth for a whole volume or z-slab.

    Proton beams in this dataset lie exactly in the transverse plane, so the
    photon slice-rotation integrator applies unchanged; slabs must span the
    full x/y extent because rays cross the entire slice.
    """
    density = proton_mass_density(
        image,
        modality=modality,
        intensity_bounds=intensity_bounds,
        synthetic_hu=synthetic_hu,
    )
    return compute_radiological_depth(
        density, geometry.spacing_xyz, gantry_angle_deg, device=device
    )


def build_range_channels(
    wepl_mm: NDArray[np.floating],
    energy_mev: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Water-equivalent depth and an analytic Bragg-peak prior.

    A convolutional network cannot integrate density along the beam, so it
    cannot know where a beamlet stops; these two channels supply that global
    quantity. ``wepl_mm`` is the accumulated water-equivalent depth, and the
    prior evaluates the Bortfeld depth-dose shape at each voxel's residual
    range, which is zero beyond the stopping point.
    """
    depth = np.asarray(wepl_mm, dtype=np.float32)
    range_mm = proton_range_mm(energy_mev)
    residual = range_mm - depth
    straggling = max(STRAGGLING_FRACTION * range_mm, 1.0)
    # tanh gives the distal falloff its finite width instead of a hard step,
    # so a small WEPL error does not flip the prior between 0 and its peak.
    falloff = 0.5 * (1.0 + np.tanh(residual / straggling))
    shape = np.power(np.maximum(residual, 1.0), BRAGG_EXPONENT)
    prior = (falloff * shape).astype(np.float32)
    normalized_depth = np.clip(depth / WEPL_NORMALIZATION_MM, 0.0, 2.0).astype(np.float32)
    return normalized_depth, np.clip(prior, 0.0, 1.0)


@dataclass(frozen=True)
class ProtonCondition:
    gantry_angle_deg: float
    ray_source_xyz: tuple[float, float, float]
    ray_target_xyz: tuple[float, float, float]
    energy_mev: float
    sigma_energy_mev: float
    sigma_spot_mm: float


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
    vx, vy, vz = xx * sx, yy * sy, zz * sz
    px = ox + direction[0, 0] * vx + direction[0, 1] * vy + direction[0, 2] * vz
    py = oy + direction[1, 0] * vx + direction[1, 1] * vy + direction[1, 2] * vz
    pz = oz + direction[2, 0] * vx + direction[2, 1] * vy + direction[2, 2] * vz
    return px, py, pz


def build_proton_channels(
    image_patch: NDArray[np.floating],
    patch_start_zyx: Sequence[int],
    geometry: SpatialGeometry,
    condition: ProtonCondition,
    *,
    modality: str,
    intensity_bounds: tuple[float, float] | None = None,
    wepl_patch: NDArray[np.floating] | None = None,
    synthetic_hu_patch: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    """Create image and pencil-beam geometry channels for one 3-D patch.

    The CT and MRI variants deliberately have the same ten-channel contract,
    allowing MRI training to warm-start from the CT model without requiring a
    CT image at MRI inference time.
    """
    image = np.asarray(image_patch, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"expected a 3-D image, got {image.shape}")
    if modality == "ct":
        low, high = (-1024.0, 2000.0)
        normalized = 2.0 * (np.clip(image, low, high) - low) / (high - low) - 1.0
        body = image > -1000.0
        density = np.clip(hu_to_mass_density(image) / 2.0, 0.0, 2.0)
    elif modality == "mri":
        low, high = intensity_bounds or (0.0, max(float(image.max()), 1.0))
        if high <= low:
            high = low + 1.0
        normalized = 2.0 * (np.clip(image, low, high) - low) / (high - low) - 1.0
        body = image > low
        if synthetic_hu_patch is None:
            density = body.astype(np.float32) * 0.5
        else:
            density = np.where(
                body,
                np.clip(
                    hu_to_mass_density(
                        np.asarray(synthetic_hu_patch, dtype=np.float32)
                    )
                    / 2.0,
                    0.0,
                    2.0,
                ),
                0.0,
            )
    else:
        raise ValueError(f"unsupported modality: {modality!r}")

    px, py, pz = _physical_coordinates(patch_start_zyx, image.shape, geometry)
    source = np.asarray(condition.ray_source_xyz, dtype=np.float32)
    target = np.asarray(condition.ray_target_xyz, dtype=np.float32)
    direction = target - source
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        raise ValueError("ray source and target must differ")
    direction /= norm

    rel_source_x, rel_source_y, rel_source_z = px - source[0], py - source[1], pz - source[2]
    along = (
        rel_source_x * direction[0]
        + rel_source_y * direction[1]
        + rel_source_z * direction[2]
    )
    closest_x = source[0] + along * direction[0]
    closest_y = source[1] + along * direction[1]
    closest_z = source[2] + along * direction[2]
    radial_sq = (px - closest_x) ** 2 + (py - closest_y) ** 2 + (pz - closest_z) ** 2
    sigma = max(float(condition.sigma_spot_mm), 1.0)
    fluence = np.exp(-0.5 * radial_sq / (sigma * sigma)).astype(np.float32)
    fluence *= (along > 0).astype(np.float32)

    # Two stable coordinates perpendicular to the ray. Proton beams in this
    # dataset rotate in the x/y plane, so z is the natural second coordinate.
    lateral_x = -direction[1]
    lateral_y = direction[0]
    rel_target_x, rel_target_y, rel_target_z = px - target[0], py - target[1], pz - target[2]
    depth_from_target = (
        rel_target_x * direction[0]
        + rel_target_y * direction[1]
        + rel_target_z * direction[2]
    )
    lateral = rel_target_x * lateral_x + rel_target_y * lateral_y
    superior = rel_target_z
    energy = 2.0 * (
        (float(condition.energy_mev) - ENERGY_MIN_MEV) / (ENERGY_MAX_MEV - ENERGY_MIN_MEV)
    ) - 1.0
    energy_map = np.full(image.shape, np.clip(energy, -1.0, 1.0), dtype=np.float32)
    energy_width = np.full(
        image.shape, np.clip(float(condition.sigma_energy_mev) / 7.0, 0.0, 1.0), dtype=np.float32
    )
    target_distance_sq = rel_target_x**2 + rel_target_y**2 + rel_target_z**2
    target_prior = np.exp(-0.5 * target_distance_sq / (50.0**2)).astype(np.float32)

    count = 10 if wepl_patch is None else 12
    channels = np.empty((count, *image.shape), dtype=np.float32)
    channels[0] = normalized
    channels[1] = body.astype(np.float32)
    channels[2] = fluence
    channels[3] = np.clip(depth_from_target / 400.0, -1.0, 1.0)
    channels[4] = np.clip(lateral / 200.0, -1.0, 1.0)
    channels[5] = np.clip(superior / 200.0, -1.0, 1.0)
    channels[6] = energy_map
    channels[7] = energy_width
    channels[8] = target_prior
    channels[9] = density
    if wepl_patch is not None:
        channels[10], channels[11] = build_range_channels(
            wepl_patch, condition.energy_mev
        )
    return channels



@dataclass(frozen=True)
class PreparedProtonPatch:
    image: NDArray[np.float32]
    normalized: NDArray[np.float32]
    body: NDArray[np.bool_]
    density: NDArray[np.float32]
    px: NDArray[np.float32]
    py: NDArray[np.float32]
    pz: NDArray[np.float32]


def prepare_proton_patch(
    image_patch: NDArray[np.floating],
    patch_start_zyx: Sequence[int],
    geometry: SpatialGeometry,
    *,
    modality: str,
    intensity_bounds: tuple[float, float] | None = None,
    synthetic_hu_patch: NDArray[np.floating] | None = None,
) -> PreparedProtonPatch:
    image = np.asarray(image_patch, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"expected a 3-D image, got {image.shape}")
    if modality == "ct":
        low, high = (-1024.0, 2000.0)
        normalized = 2.0 * (np.clip(image, low, high) - low) / (high - low) - 1.0
        body = image > -1000.0
        density = np.clip(hu_to_mass_density(image) / 2.0, 0.0, 2.0)
    elif modality == "mri":
        low, high = intensity_bounds or (0.0, max(float(image.max()), 1.0))
        if high <= low:
            high = low + 1.0
        normalized = 2.0 * (np.clip(image, low, high) - low) / (high - low) - 1.0
        body = image > low
        if synthetic_hu_patch is None:
            density = body.astype(np.float32) * 0.5
        else:
            density = np.where(
                body,
                np.clip(
                    hu_to_mass_density(
                        np.asarray(synthetic_hu_patch, dtype=np.float32)
                    )
                    / 2.0,
                    0.0,
                    2.0,
                ),
                0.0,
            )
    else:
        raise ValueError(f"unsupported modality: {modality!r}")
    px, py, pz = _physical_coordinates(patch_start_zyx, image.shape, geometry)
    return PreparedProtonPatch(
        image=image,
        normalized=np.asarray(normalized, dtype=np.float32),
        body=body,
        density=np.asarray(density, dtype=np.float32),
        px=px,
        py=py,
        pz=pz,
    )


def build_proton_channels_batch(
    image_patch: NDArray[np.floating],
    patch_start_zyx: Sequence[int],
    geometry: SpatialGeometry,
    conditions: Sequence[ProtonCondition],
    *,
    modality: str,
    intensity_bounds: tuple[float, float] | None = None,
    prepared: PreparedProtonPatch | None = None,
    wepl_patch: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    """Create several condition tensors while reusing static and ray work."""
    if not conditions:
        raise ValueError("conditions must not be empty")
    context = prepared or prepare_proton_patch(
        image_patch,
        patch_start_zyx,
        geometry,
        modality=modality,
        intensity_bounds=intensity_bounds,
    )
    image = context.image
    px, py, pz = context.px, context.py, context.pz
    count = 10 if wepl_patch is None else 12
    channels = np.empty((len(conditions), count, *image.shape), dtype=np.float32)
    channels[:, 0] = context.normalized
    channels[:, 1] = context.body.astype(np.float32)
    channels[:, 9] = context.density
    ray_cache: dict[tuple[tuple[float, ...], tuple[float, ...]], tuple[NDArray[np.float32], ...]] = {}

    for index, condition in enumerate(conditions):
        key = (condition.ray_source_xyz, condition.ray_target_xyz)
        cached = ray_cache.get(key)
        if cached is None:
            source = np.asarray(condition.ray_source_xyz, dtype=np.float32)
            target = np.asarray(condition.ray_target_xyz, dtype=np.float32)
            direction = target - source
            norm = float(np.linalg.norm(direction))
            if norm <= 0:
                raise ValueError("ray source and target must differ")
            direction /= norm

            rel_source_x = px - source[0]
            rel_source_y = py - source[1]
            rel_source_z = pz - source[2]
            along = (
                rel_source_x * direction[0]
                + rel_source_y * direction[1]
                + rel_source_z * direction[2]
            )
            closest_x = source[0] + along * direction[0]
            closest_y = source[1] + along * direction[1]
            closest_z = source[2] + along * direction[2]
            radial_sq = (
                (px - closest_x) ** 2
                + (py - closest_y) ** 2
                + (pz - closest_z) ** 2
            )
            rel_target_x = px - target[0]
            rel_target_y = py - target[1]
            rel_target_z = pz - target[2]
            depth_from_target = (
                rel_target_x * direction[0]
                + rel_target_y * direction[1]
                + rel_target_z * direction[2]
            )
            lateral = (
                rel_target_x * -direction[1]
                + rel_target_y * direction[0]
            )
            target_distance_sq = (
                rel_target_x**2 + rel_target_y**2 + rel_target_z**2
            )
            target_prior = np.exp(
                -0.5 * target_distance_sq / (50.0**2)
            ).astype(np.float32)
            cached = (
                radial_sq,
                along,
                depth_from_target,
                lateral,
                rel_target_z,
                target_prior,
            )
            ray_cache[key] = cached
        radial_sq, along, depth_from_target, lateral, superior, target_prior = cached

        sigma = max(float(condition.sigma_spot_mm), 1.0)
        fluence = np.exp(-0.5 * radial_sq / (sigma * sigma)).astype(np.float32)
        fluence *= (along > 0).astype(np.float32)
        energy = 2.0 * (
            (float(condition.energy_mev) - ENERGY_MIN_MEV)
            / (ENERGY_MAX_MEV - ENERGY_MIN_MEV)
        ) - 1.0
        channels[index, 2] = fluence
        channels[index, 3] = np.clip(depth_from_target / 400.0, -1.0, 1.0)
        channels[index, 4] = np.clip(lateral / 200.0, -1.0, 1.0)
        channels[index, 5] = np.clip(superior / 200.0, -1.0, 1.0)
        channels[index, 6].fill(float(np.clip(energy, -1.0, 1.0)))
        channels[index, 7].fill(
            float(np.clip(float(condition.sigma_energy_mev) / 7.0, 0.0, 1.0))
        )
        channels[index, 8] = target_prior
        if wepl_patch is not None:
            channels[index, 10], channels[index, 11] = build_range_channels(
                wepl_patch, condition.energy_mev
            )
    return channels
