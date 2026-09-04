from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from doserad_photon_mri.conditioning import (  # noqa: E402
    PhotonCondition,
    SpatialGeometry,
    build_condition_channels,
    mri_foreground_bounds,
)
from doserad_photon_mri.inference import (  # noqa: E402
    predict_conditioned_array,
    predict_conditioned_arrays,
)
from doserad_photon_mri.losses import DoseLoss, LossConfig  # noqa: E402
from doserad_photon_mri.model import ModelConfig, PhotonDoseUNet3D  # noqa: E402


def open_condition(angle: float = 0.0) -> PhotonCondition:
    return PhotonCondition(
        gantry_angle_deg=angle,
        iso_center_xyz=(7.5, 7.5, 7.5),
        mlc_left_int_mm=tuple([-19.5] * 8),
        mlc_right_int_mm=tuple([20.5] * 8),
    )


def geometry() -> SpatialGeometry:
    return SpatialGeometry(
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )


class ConstantDoseModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (inputs.shape[0], 1, *inputs.shape[2:]),
            2.0,
            dtype=inputs.dtype,
            device=inputs.device,
        )


class PhotonMRIPipelineTest(unittest.TestCase):
    def test_mri_scaling_ignores_zero_background(self) -> None:
        image = np.zeros((10, 10, 10), dtype=np.float32)
        image[2:8, 2:8, 2:8] = np.linspace(10.0, 300.0, 6**3).reshape(6, 6, 6)
        low, high = mri_foreground_bounds(image)
        self.assertGreater(low, 0.0)
        self.assertGreater(high, low)

    def test_channels_have_mri_body_and_aperture(self) -> None:
        image = np.zeros((16, 16, 16), dtype=np.float32)
        image[2:14, 2:14, 2:14] = 100.0
        channels = build_condition_channels(
            image, (0, 0, 0), geometry(), open_condition(),
            intensity_bounds=(1.0, 200.0),
        )
        self.assertEqual(channels.shape, (6, 16, 16, 16))
        self.assertTrue(np.isfinite(channels).all())
        self.assertEqual(float(channels[1, 0, 0, 0]), 0.0)
        self.assertEqual(float(channels[1, 8, 8, 8]), 1.0)
        self.assertGreater(float(channels[2].mean()), 0.9)

    def test_physics_prior_channels_are_finite(self) -> None:
        image = np.zeros((16, 16, 16), dtype=np.float32)
        image[2:14, 2:14, 2:14] = 100.0
        channels = build_condition_channels(
            image,
            (0, 0, 0),
            geometry(),
            open_condition(),
            intensity_bounds=(1.0, 200.0),
            include_physics_priors=True,
        )
        self.assertEqual(channels.shape, (10, 16, 16, 16))
        self.assertTrue(np.isfinite(channels).all())
        self.assertGreater(float(channels[8].mean()), 0.0)
        self.assertGreater(float(channels[9].mean()), 0.0)

    def test_auxiliary_density_head_and_competition_losses(self) -> None:
        model = PhotonDoseUNet3D(
            ModelConfig(
                in_channels=10,
                base_channels=4,
                levels=2,
                blocks_per_level=1,
                physics_priors=True,
                auxiliary_density=True,
            )
        )
        inputs = torch.randn(2, 10, 8, 8, 8)
        dose, density = model.forward_with_auxiliary(inputs)
        self.assertEqual(tuple(dose.shape), (2, 1, 8, 8, 8))
        self.assertIsNotNone(density)
        self.assertEqual(tuple(density.shape), (2, 1, 8, 8, 8))
        target = torch.rand_like(dose)
        loss, components = DoseLoss(
            LossConfig(
                official_mae_weight=1.0,
                idd_surrogate_weight=0.1,
                scale_weight=0.1,
            )
        )(dose, target, target.amax(dim=(1, 2, 3, 4)), None, torch.tensor([0.0, 45.0]))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(components),
            {"full", "high", "gradient", "official_mae", "idd_surrogate", "scale"},
        )

    def test_inference_restores_shape_and_masks_mri_background(self) -> None:
        image = np.zeros((7, 8, 9), dtype=np.float32)
        image[1:-1, 1:-1, 1:-1] = 100.0
        prediction = predict_conditioned_array(
            ConstantDoseModel(), image=image, geometry=geometry(),
            condition=open_condition(), device=torch.device("cpu"),
            patch_size_zyx=(10, 10, 10), dose_scale=1.0e-4,
            overlap=0.5, batch_size=2, amp=False, mask_outside_body=True,
        )
        self.assertEqual(prediction.shape, image.shape)
        self.assertEqual(float(prediction[0, 0, 0]), 0.0)
        np.testing.assert_allclose(prediction[1:-1, 1:-1, 1:-1], 2.0e-4)

    def test_batched_control_points_match_sequential(self) -> None:
        image = np.ones((7, 8, 9), dtype=np.float32) * 100.0
        conditions = [open_condition(), open_condition(45.0)]
        model = torch.nn.Sequential(torch.nn.Conv3d(6, 1, 1), torch.nn.Softplus())
        common = dict(
            model=model, image=image, geometry=geometry(), device=torch.device("cpu"),
            patch_size_zyx=(6, 6, 6), dose_scale=1.0, overlap=0.25, amp=False,
        )
        sequential = [
            predict_conditioned_array(condition=condition, batch_size=2, **common)
            for condition in conditions
        ]
        batched = predict_conditioned_arrays(
            conditions=conditions, condition_batch_size=2, **common
        )
        np.testing.assert_allclose(np.stack(batched), np.stack(sequential), rtol=1e-6)

    def test_padded_inference_batch_matches_unpadded(self) -> None:
        image = np.ones((7, 8, 9), dtype=np.float32) * 100.0
        conditions = [open_condition(), open_condition(45.0), open_condition(90.0)]
        model = torch.nn.Sequential(torch.nn.Conv3d(6, 1, 1), torch.nn.Softplus())
        common = dict(
            model=model,
            image=image,
            geometry=geometry(),
            conditions=conditions,
            device=torch.device("cpu"),
            patch_size_zyx=(6, 6, 6),
            dose_scale=1.0,
            overlap=0.25,
            condition_batch_size=2,
            amp=False,
        )
        unpadded = predict_conditioned_arrays(**common)
        padded = predict_conditioned_arrays(**common, pad_to_batch_size=True)
        np.testing.assert_allclose(np.stack(padded), np.stack(unpadded), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
