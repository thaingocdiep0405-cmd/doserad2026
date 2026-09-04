"""Analytic proton pencil-beam dose via pyRadPlan.

The challenge ships a pyRadPlan pencil-beam baseline, and on validation
beamlets that engine is far closer to the Monte-Carlo reference than the
trained network: masked MAE 0.028 against 0.052, and IDD 0.0096 against 0.130.
It needs mass density, which MRI does not carry, so the body is substituted
with water — the same first-order assumption the network's density channel
already makes, and it costs only about 0.005 of MAE.

Dose is computed on a slab of transverse slices rather than the whole grid.
Protons in this dataset travel exactly in the transverse plane, so slices away
from the beam carry nothing and dropping them is free.

The transverse plane is kept whole. An earlier version also cropped x and y to
a corridor around the ray, which is what made thoracic beamlets fail: on 23
thoracic beamlets against the real CT the corridor kept only 71% of the dose
and 12 of them lost most of it, for a mean IDD of 0.1348. Keeping the full
plane and cropping z alone retains 100% of the dose on every one of them and
brings mean IDD to 0.0175, for 2.1s a beamlet against 1.0s. Doubling the slab
to 24 half-slices changes nothing (same IDD to four decimals, 3.9s), so 12 is
where the slab stops paying.

The corridor was not merely too small -- widening it to a 120 mm margin still
recovered only 15% of the dose on the worst beamlet while the full grid
recovered 104%. Cropping the transverse plane cuts through the body, and both
the body segmentation and the air-offset correction are rebuilt from that
sliced CT, so the depth the protons are assumed to have travelled is wrong.
"""
from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

WATER_HU = 0.0
AIR_HU = -1000.0
SLAB_HALF_SLICES = 12
LATERAL_CUTOFF_MM = 25.0
# In-plane dose-grid resolution for the batched path. Per-beamlet time in a
# batch is dominated by per-beam work that scales with dose-grid voxels, and
# 2 mm in-plane measured both faster and closer to the Monte-Carlo reference
# than the native 1 mm grid (MAE 0.0342 vs 0.0379, IDD 0.0150 vs 0.0212 over
# 20 thoracic beamlets); 3 mm was faster still but gave back accuracy. The
# z resolution stays at the CT's own slice spacing.
BATCH_LATERAL_RESOLUTION_MM = 2.0


def _patch_machine_cache() -> None:
    """Wrap pyRadPlan's machine loader with lru_cache so the .mat file is
    read once per (radiation_mode, machine_name) instead of every beamlet."""
    from pyRadPlan.dose.engines import _base as eng_base

    original_fn = eng_base.DoseEngineBase.load_machine
    if getattr(original_fn, "_cached", False):
        return

    cached_fn = functools.lru_cache(maxsize=4)(original_fn)
    cached_fn._cached = True
    eng_base.DoseEngineBase.load_machine = staticmethod(cached_fn)


_patch_machine_cache()


@dataclass(frozen=True)
class BeamletSpec:
    """Everything pyRadPlan needs for one beamlet, taken from the plan JSON."""

    gantry_angle_deg: float
    ray_source_xyz: tuple[float, float, float]
    ray_target_xyz: tuple[float, float, float]
    energy_mev: float


class PencilBeamEngine:
    """Reusable pyRadPlan forward dose engine for single beamlets."""

    def __init__(self, beam_parameters_path: Path | str) -> None:
        from pyRadPlan.machines import load_from_name

        payload = json.loads(Path(beam_parameters_path).read_text(encoding="utf-8"))
        entries = payload["hu_to_density"]["entries"]
        self.hlut = np.array([tuple(row.values()) for row in entries], dtype=float)
        machine = load_from_name("protons", "Generic")
        self.energies = np.asarray(sorted(machine.energies), dtype=np.float64)
        self.sad = float(machine.sad)
        self.lateral_cutoff_mm = LATERAL_CUTOFF_MM
        self.slab_half_slices = SLAB_HALF_SLICES

    def snap_energy(self, energy_mev: float) -> float:
        """Nearest energy present in the machine lookup table."""
        return float(self.energies[int(np.argmin(np.abs(self.energies - energy_mev)))])

    @staticmethod
    def pseudo_ct_from_mri(
        image: NDArray[np.floating], body_threshold: float
    ) -> NDArray[np.float32]:
        """Water inside the body, air outside, in Hounsfield units."""
        values = np.asarray(image, dtype=np.float32)
        return np.where(values > body_threshold, WATER_HU, AIR_HU).astype(np.float32)

    def corridor_box(
        self,
        shape_zyx: tuple[int, int, int],
        origin_xyz: NDArray[np.floating],
        spacing_xyz: NDArray[np.floating],
        spec: BeamletSpec,
    ) -> tuple[int, int, int, int, int, int]:
        """Index bounds (z0, z1, y0, y1, x0, x1) covering the beamlet corridor."""
        target = np.asarray(spec.ray_target_xyz, dtype=float)
        source = np.asarray(spec.ray_source_xyz, dtype=float)
        if float(np.linalg.norm(target - source)) <= 0:
            raise ValueError("ray source and target must differ")
        centre_z = int(round((target[2] - origin_xyz[2]) / spacing_xyz[2]))
        z0 = max(0, centre_z - self.slab_half_slices)
        z1 = min(int(shape_zyx[0]), centre_z + self.slab_half_slices)
        if z1 <= z0:
            raise ValueError("empty corridor box")
        return z0, z1, 0, int(shape_zyx[1]), 0, int(shape_zyx[2])

    @staticmethod
    def _create_ct(
        hu_zyx: NDArray[np.floating],
        origin_xyz: tuple[float, float, float],
        spacing_xyz: tuple[float, float, float],
        direction: tuple[float, ...],
    ):
        from pyRadPlan.ct import create_ct

        image = sitk.GetImageFromArray(np.ascontiguousarray(hu_zyx, dtype=np.float32))
        image.SetSpacing(spacing_xyz)
        image.SetDirection(direction)
        image.SetOrigin(origin_xyz)
        return create_ct(cube_hu=image)

    def dose(
        self,
        hu_zyx: NDArray[np.floating],
        reference: sitk.Image,
        spec: BeamletSpec,
    ) -> NDArray[np.float32]:
        """Pencil-beam dose for one beamlet on the reference grid."""
        z0, patch = self.dose_slab(
            hu_zyx,
            reference.GetOrigin(),
            reference.GetSpacing(),
            reference.GetDirection(),
            spec,
        )
        out = np.zeros(tuple(int(v) for v in np.asarray(hu_zyx).shape), dtype=np.float32)
        out[z0 : z0 + patch.shape[0]] = patch
        return out

    def dose_slab(
        self,
        hu_zyx: NDArray[np.floating],
        origin_xyz: tuple[float, ...],
        spacing_xyz: tuple[float, ...],
        direction: tuple[float, ...],
        spec: BeamletSpec,
    ) -> tuple[int, NDArray[np.float32]]:
        """Dose for one beamlet as (first slice, slab) instead of a full volume."""
        from pyRadPlan import IonPlan, calc_dose_forward
        from pyRadPlan.cst import StructureSet
        from pyRadPlan.geometry import get_beam_rotation_matrix
        from pyRadPlan.stf import SteeringInformation
        from pyRadPlan.stf._beam import Beam

        origin = np.asarray(origin_xyz, dtype=float)
        spacing = np.asarray(spacing_xyz, dtype=float)
        shape = tuple(int(v) for v in np.asarray(hu_zyx).shape)
        z0, z1, y0, y1, x0, x1 = self.corridor_box(shape, origin, spacing, spec)
        sub = np.asarray(hu_zyx, dtype=np.float32)[z0:z1, y0:y1, x0:x1]
        sub_origin = (
            float(origin[0] + x0 * spacing[0]),
            float(origin[1] + y0 * spacing[1]),
            float(origin[2] + z0 * spacing[2]),
        )
        ct = self._create_ct(
            sub, sub_origin, tuple(float(v) for v in spacing), direction
        )

        cst = StructureSet(vois=[], ct_image=ct)
        cst.create_body_seg()
        plan = IonPlan(radiation_mode="protons", machine="Generic")
        plan.prop_dose_calc = {
            "dose_grid": {"resolution": ct.resolution},
            "air_offset_correction": True,
            "geometric_lateral_cutoff": self.lateral_cutoff_mm,
            "trace_on_dose_grid": True,
            "hlut": self.hlut,
            "calc_let": False,
        }
        rotation = get_beam_rotation_matrix(float(spec.gantry_angle_deg), 0.0)
        source_bev = np.array([0.0, -self.sad, 0.0])
        beam = Beam.model_validate(
            {
                "gantry_angle": float(spec.gantry_angle_deg),
                "couch_angle": 0.0,
                "radiation_mode": "protons",
                "machine": "Generic",
                "SAD": self.sad,
                "iso_center": np.asarray(spec.ray_target_xyz, dtype=float),
                "source_point_bev": source_bev,
                "source_point": rotation @ source_bev,
                "rays": [
                    {
                        "ray_pos_bev": np.zeros(3),
                        "ray_pos": np.zeros(3),
                        "beamlets": [{"energy": self.snap_energy(spec.energy_mev)}],
                    }
                ],
            }
        )
        result = calc_dose_forward(ct, cst, SteeringInformation(beams=[beam]), plan)
        patch = sitk.GetArrayFromImage(result["physical_dose"]).astype(np.float32)
        slab = np.zeros((z1 - z0, shape[1], shape[2]), dtype=np.float32)
        slab[: patch.shape[0], y0 : y0 + patch.shape[1], x0 : x0 + patch.shape[2]] = patch
        return z0, slab

    def _make_beam(self, spec: BeamletSpec):
        from pyRadPlan.geometry import get_beam_rotation_matrix
        from pyRadPlan.stf._beam import Beam

        rotation = get_beam_rotation_matrix(float(spec.gantry_angle_deg), 0.0)
        source_bev = np.array([0.0, -self.sad, 0.0])
        return Beam.model_validate(
            {
                "gantry_angle": float(spec.gantry_angle_deg),
                "couch_angle": 0.0,
                "radiation_mode": "protons",
                "machine": "Generic",
                "SAD": self.sad,
                "iso_center": np.asarray(spec.ray_target_xyz, dtype=float),
                "source_point_bev": source_bev,
                "source_point": rotation @ source_bev,
                "rays": [
                    {
                        "ray_pos_bev": np.zeros(3),
                        "ray_pos": np.zeros(3),
                        "beamlets": [{"energy": self.snap_energy(spec.energy_mev)}],
                    }
                ],
            }
        )

    def dose_slab_batch(
        self,
        hu_zyx: NDArray[np.floating],
        origin_xyz: tuple[float, ...],
        spacing_xyz: tuple[float, ...],
        direction: tuple[float, ...],
        specs: list[BeamletSpec],
    ) -> list[tuple[int, NDArray[np.float32]]]:
        """Dose for beamlets sharing one z-slab, via a single influence call.

        The per-beamlet cost of dose_slab is mostly per-call overhead: body
        segmentation, engine and machine initialisation, and grid resampling
        are all repeated for every beamlet even though every beamlet in a plan
        shares a handful of isocenter slices. Computing the whole group as one
        influence matrix pays that overhead once; each beamlet then only costs
        its own raytrace and kernel superposition. Extraction goes through
        Dij.compute_result_ct_grid with a one-hot weight vector, which is the
        exact resampling path calc_dose_forward uses.

        All specs must share the same slab window (same isocenter slice).
        Returns one (z0, slab) pair per spec, in input order.
        """
        from pyRadPlan import IonPlan
        from pyRadPlan.dose import calc_dose_influence
        from pyRadPlan.cst import StructureSet
        from pyRadPlan.stf import SteeringInformation

        origin = np.asarray(origin_xyz, dtype=float)
        spacing = np.asarray(spacing_xyz, dtype=float)
        shape = tuple(int(v) for v in np.asarray(hu_zyx).shape)
        boxes = [self.corridor_box(shape, origin, spacing, spec) for spec in specs]
        z0 = min(box[0] for box in boxes)
        z1 = max(box[1] for box in boxes)
        sub = np.asarray(hu_zyx, dtype=np.float32)[z0:z1]
        sub_origin = (float(origin[0]), float(origin[1]), float(origin[2] + z0 * spacing[2]))
        ct = self._create_ct(
            sub, sub_origin, tuple(float(v) for v in spacing), direction
        )

        cst = StructureSet(vois=[], ct_image=ct)
        cst.create_body_seg()
        plan = IonPlan(radiation_mode="protons", machine="Generic")
        resolution = dict(ct.resolution)
        resolution["x"] = max(float(resolution["x"]), BATCH_LATERAL_RESOLUTION_MM)
        resolution["y"] = max(float(resolution["y"]), BATCH_LATERAL_RESOLUTION_MM)
        plan.prop_dose_calc = {
            "dose_grid": {"resolution": resolution},
            "air_offset_correction": True,
            "geometric_lateral_cutoff": self.lateral_cutoff_mm,
            "trace_on_dose_grid": True,
            "hlut": self.hlut,
            "calc_let": False,
        }
        stf = SteeringInformation(beams=[self._make_beam(spec) for spec in specs])
        dij = calc_dose_influence(ct, cst, stf, plan)

        out: list[tuple[int, NDArray[np.float32]]] = []
        weights = np.zeros(len(specs), dtype=np.float32)
        for index in range(len(specs)):
            weights[index] = 1.0
            result = dij.compute_result_ct_grid(weights)
            weights[index] = 0.0
            patch = sitk.GetArrayFromImage(result["physical_dose"]).astype(np.float32)
            slab = np.zeros((z1 - z0, shape[1], shape[2]), dtype=np.float32)
            slab[: patch.shape[0], : patch.shape[1], : patch.shape[2]] = patch
            out.append((z0, slab))
        return out
