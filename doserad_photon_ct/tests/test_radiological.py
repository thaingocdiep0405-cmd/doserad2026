import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from doserad_photon_ct.conditioning import (  # noqa: E402
    PhotonCondition,
    SpatialGeometry,
    build_condition_channels,
)
from doserad_photon_ct.radiological import compute_radiological_depth  # noqa: E402

SPACING = (2.0, 2.0, 2.0)


def open_condition(gantry: float = 0.0) -> PhotonCondition:
    return PhotonCondition(
        gantry_angle_deg=gantry,
        iso_center_xyz=(0.0, 0.0, 0.0),
        mlc_left_int_mm=tuple([-100.0] * 40),
        mlc_right_int_mm=tuple([100.0] * 40),
    )


class RadiologicalDepthTest(unittest.TestCase):
    def test_axis_aligned_gantries_match_direct_integration(self) -> None:
        shape = (3, 48, 40)
        density = np.ones(shape, dtype=np.float32)
        density[:, 12:20, :] = 2.0
        density[:, :, 8:16] = 1.5
        cases = {
            0.0: lambda: (np.cumsum(density, axis=1) - 0.5 * density) * 2.0,
            180.0: lambda: np.flip(
                (np.cumsum(np.flip(density, 1), axis=1) - 0.5 * np.flip(density, 1))
                * 2.0,
                1,
            ),
            90.0: lambda: np.flip(
                (np.cumsum(np.flip(density, 2), axis=2) - 0.5 * np.flip(density, 2))
                * 2.0,
                2,
            ),
            270.0: lambda: (np.cumsum(density, axis=2) - 0.5 * density) * 2.0,
        }
        for gantry, direct in cases.items():
            with self.subTest(gantry=gantry):
                depth = compute_radiological_depth(
                    density, SPACING, gantry, downsample_xy=1
                )
                expected = direct()
                interior = np.abs(depth[:, 4:-4, 4:-4] - expected[:, 4:-4, 4:-4])
                self.assertLess(float(interior.max()), 0.1)

    def test_downsampled_depth_tracks_smooth_density(self) -> None:
        # The production default (downsample_xy=2) trades sharp-edge accuracy
        # for speed; on smooth density it must stay within ~1 voxel of water.
        yy = np.linspace(0.0, 1.0, 48, dtype=np.float32)
        xx = np.linspace(0.0, 1.0, 40, dtype=np.float32)
        density = np.broadcast_to(
            1.0 + 0.4 * np.sin(3.0 * yy)[:, None] * np.cos(2.0 * xx)[None, :],
            (3, 48, 40),
        ).astype(np.float32)
        expected = (np.cumsum(density, axis=1) - 0.5 * density) * 2.0
        depth = compute_radiological_depth(density, SPACING, 0.0, downsample_xy=2)
        interior = np.abs(depth[:, 4:-4, 4:-4] - expected[:, 4:-4, 4:-4])
        self.assertLess(float(interior.max()), 2.0)

    def test_oblique_gantry_is_finite_monotonic_along_beam(self) -> None:
        density = np.ones((2, 60, 60), dtype=np.float32)
        depth = compute_radiological_depth(density, SPACING, 45.0)
        self.assertTrue(np.isfinite(depth).all())
        self.assertGreaterEqual(float(depth.min()), 0.0)
        gantry = math.radians(45.0)
        # Step one voxel along the beam (dy, dx) = (cos g, -sin g): depth
        # must increase by about one diagonal step of water.
        y, x = 30, 30
        step_y = int(round(math.cos(gantry) * 8))
        step_x = int(round(-math.sin(gantry) * 8))
        difference = float(depth[0, y + step_y, x + step_x] - depth[0, y, x])
        expected = math.hypot(step_y, step_x) * 2.0
        self.assertAlmostEqual(difference, expected, delta=3.0)

    def test_conditioning_appends_two_channels(self) -> None:
        ct = np.zeros((8, 8, 8), dtype=np.float32)
        geometry = SpatialGeometry(
            spacing_xyz=SPACING, origin_xyz=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        rad = np.full((8, 8, 8), 100.0, dtype=np.float32)
        channels = build_condition_channels(
            ct,
            (0, 0, 0),
            geometry,
            open_condition(),
            include_physics_priors=True,
            radiological_depth_patch=rad,
        )
        self.assertEqual(channels.shape[0], 13)
        self.assertTrue(np.isfinite(channels).all())
        np.testing.assert_allclose(channels[11], 100.0 / 400.0)
        # Attenuated fluence must stay below the unattenuated prior.
        self.assertTrue(np.all(channels[12] <= channels[10] + 1.0e-6))

    def test_radiological_channels_require_physics_priors(self) -> None:
        ct = np.zeros((4, 4, 4), dtype=np.float32)
        geometry = SpatialGeometry(
            spacing_xyz=SPACING, origin_xyz=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        with self.assertRaises(ValueError):
            build_condition_channels(
                ct,
                (0, 0, 0),
                geometry,
                open_condition(),
                radiological_depth_patch=np.zeros((4, 4, 4), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
