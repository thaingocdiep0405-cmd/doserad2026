from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from doserad_photon_ct.conditioning import (  # noqa: E402
    PhotonCondition,
    SpatialGeometry,
    build_condition_channels,
    hu_to_mass_density,
)
from doserad_photon_ct.dataset import PhotonCTPatchDataset  # noqa: E402
from doserad_photon_ct.inference import (  # noqa: E402
    predict_conditioned_array,
    predict_conditioned_arrays,
)
from doserad_photon_ct.losses import DoseLoss, LossConfig  # noqa: E402
from doserad_photon_ct.mha import load_mha_array, write_mha_array_like  # noqa: E402
from doserad_photon_ct.metrics import (  # noqa: E402
    beam_direction,
    idd_curve_distance,
)
from doserad_photon_ct.model import ModelConfig, PhotonDoseUNet3D  # noqa: E402



def write_test_mha(path: Path, array: np.ndarray, *, compress: bool = True) -> None:
    values = np.asarray(array, dtype="<f4", order="C")
    payload = values.tobytes(order="C")
    if compress:
        payload = zlib.compress(payload)
    z, y, x = values.shape
    lines = [
        "ObjectType = Image",
        "NDims = 3",
        "BinaryData = True",
        "BinaryDataByteOrderMSB = False",
        f"CompressedData = {'True' if compress else 'False'}",
        "TransformMatrix = 1 0 0 0 1 0 0 0 1",
        "Offset = 0 0 0",
        "ElementSpacing = 1 1 1",
        f"DimSize = {x} {y} {z}",
        "ElementType = MET_FLOAT",
        "ElementDataFile = LOCAL",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("ascii") + payload)


def open_condition() -> PhotonCondition:
    return PhotonCondition(
        gantry_angle_deg=0.0,
        iso_center_xyz=(7.5, 7.5, 7.5),
        mlc_left_int_mm=tuple([-19.5] * 8),
        mlc_right_int_mm=tuple([20.5] * 8),
        sad_mm=1000.0,
    )


class ConstantDoseModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (inputs.shape[0], 1, *inputs.shape[2:]),
            2.0,
            dtype=inputs.dtype,
            device=inputs.device,
        )


class TrainingPipelineTest(unittest.TestCase):
    def test_official_hu_density_calibration(self) -> None:
        hu = np.asarray([-1024.0, -200.0, -10.0, 120.0, 3000.0], dtype=np.float32)
        density = hu_to_mass_density(hu)
        np.testing.assert_allclose(
            density,
            [0.0012, 0.8043754, 1.006579, 1.126553, 3.027294],
            rtol=1.0e-6,
        )

    def test_idd_is_zero_for_identical_dose_and_positive_for_shift(self) -> None:
        target = np.zeros((5, 9, 9), dtype=np.float32)
        target[:, 2:7, 3:6] = np.linspace(0.1, 1.0, 5)[:, None, None]
        direction = beam_direction(0.0)
        self.assertAlmostEqual(
            idd_curve_distance(target, target, direction, (2.0, 2.0, 2.0)), 0.0
        )
        shifted = np.roll(target, 2, axis=1)
        self.assertGreater(
            idd_curve_distance(shifted, target, direction, (2.0, 2.0, 2.0)), 0.0
        )

    def test_conditioning_channels_are_finite_and_include_aperture(self) -> None:
        ct = np.zeros((16, 16, 16), dtype=np.float32)
        geometry = SpatialGeometry(
            spacing_xyz=(1.0, 1.0, 1.0),
            origin_xyz=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        channels = build_condition_channels(ct, (0, 0, 0), geometry, open_condition())

        self.assertEqual(channels.shape, (6, 16, 16, 16))
        self.assertTrue(np.isfinite(channels).all())
        self.assertGreater(float(channels[2].mean()), 0.9)
        self.assertTrue(np.all(channels[1] == 1.0))

        density_channels = build_condition_channels(
            ct,
            (0, 0, 0),
            geometry,
            open_condition(),
            include_density=True,
        )
        self.assertEqual(density_channels.shape, (7, 16, 16, 16))

        physics_channels = build_condition_channels(
            ct,
            (0, 0, 0),
            geometry,
            open_condition(),
            include_physics_priors=True,
        )
        self.assertEqual(physics_channels.shape, (11, 16, 16, 16))
        self.assertTrue(np.isfinite(physics_channels).all())
        self.assertGreater(float(physics_channels[10].mean()), 0.0)
        self.assertTrue(np.all(physics_channels[8] >= 0.0))

    def test_mha_round_trip_preserves_float_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.mha"
            output = root / "output.mha"
            source = np.arange(8 * 7 * 6, dtype=np.float32).reshape(8, 7, 6)
            prediction = source / 100.0
            write_test_mha(reference, source)

            write_mha_array_like(output, prediction, reference, compress=True)

            np.testing.assert_allclose(load_mha_array(output), prediction)

    def test_patch_dataset_model_and_loss_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            patient = root / "photon" / "training" / "1ABB001"
            ct_path = patient / "image" / "ct.mha"
            dose_path = patient / "dose" / "Dose_B0_CP000.mha"
            metadata_path = patient / "1ABB001.json"
            manifest_path = root / "manifest.csv"
            splits_path = root / "splits.json"

            ct = np.zeros((16, 16, 16), dtype=np.float32)
            dose = np.zeros_like(ct)
            dose[6:10, 6:10, 6:10] = 1.0e-4
            write_test_mha(ct_path, ct)
            write_test_mha(dose_path, dose)
            condition = open_condition()
            metadata = {
                "beams": [
                    {
                        "beam_idx": 0,
                        "SAD": condition.sad_mm,
                        "iso_center": list(condition.iso_center_xyz),
                        "num_mlc_leaf_pairs": condition.leaf_pairs,
                        "control_points": [
                            {
                                "cp_idx": 0,
                                "gantry_angle": condition.gantry_angle_deg,
                                "mlc_left_int_mm": list(condition.mlc_left_int_mm),
                                "mlc_right_int_mm": list(condition.mlc_right_int_mm),
                            }
                        ],
                    }
                ]
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "patient_id",
                        "anatomy_group",
                        "ct_path",
                        "metadata_path",
                        "dose_path",
                        "beam_idx",
                        "cp_idx",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "patient_id": "1ABB001",
                        "anatomy_group": "ABB",
                        "ct_path": ct_path,
                        "metadata_path": metadata_path,
                        "dose_path": dose_path,
                        "beam_idx": 0,
                        "cp_idx": 0,
                    }
                )
            splits_path.write_text(
                json.dumps({"train": ["1ABB001"], "validation": ["1ABB001"]}),
                encoding="utf-8",
            )

            dataset = PhotonCTPatchDataset(
                manifest_path,
                splits_path,
                "train",
                patch_size_zyx=(8, 8, 8),
                positive_patch_probability=1.0,
                ct_cache_size=1,
            )
            sample = dataset[0]
            model = PhotonDoseUNet3D(
                ModelConfig(base_channels=4, levels=3, blocks_per_level=1)
            )
            prediction = model(sample["input"].unsqueeze(0))
            loss, components = DoseLoss(LossConfig())(
                prediction,
                sample["target"].unsqueeze(0),
                gantry_angles_deg=torch.tensor([37.0]),
            )
            loss.backward()

            self.assertEqual(tuple(sample["input"].shape), (6, 8, 8, 8))
            self.assertEqual(tuple(prediction.shape), (1, 1, 8, 8, 8))
            self.assertGreater(float(sample["target"].max()), 0.0)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(
                set(components),
                {"full", "high", "gradient", "official_mae", "idd_surrogate", "scale",
                 "out_of_field"},
            )
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_sliding_window_inference_restores_original_shape_and_scale(self) -> None:
        ct = np.zeros((7, 8, 9), dtype=np.float32)
        geometry = SpatialGeometry(
            spacing_xyz=(1.0, 1.0, 1.0),
            origin_xyz=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        prediction = predict_conditioned_array(
            ConstantDoseModel(),
            ct=ct,
            geometry=geometry,
            condition=open_condition(),
            device=torch.device("cpu"),
            patch_size_zyx=(10, 10, 10),
            dose_scale=1.0e-4,
            overlap=0.5,
            batch_size=2,
            amp=False,
        )

        self.assertEqual(prediction.shape, ct.shape)
        np.testing.assert_allclose(prediction, 2.0e-4, rtol=1.0e-5, atol=1.0e-8)

    def test_batched_control_points_match_sequential_inference(self) -> None:
        ct = np.zeros((7, 8, 9), dtype=np.float32)
        geometry = SpatialGeometry(
            spacing_xyz=(1.0, 1.0, 1.0),
            origin_xyz=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        conditions = [
            open_condition(),
            PhotonCondition(
                gantry_angle_deg=45.0,
                iso_center_xyz=(7.5, 7.5, 7.5),
                mlc_left_int_mm=tuple([-19.5] * 8),
                mlc_right_int_mm=tuple([20.5] * 8),
            ),
        ]
        model = torch.nn.Sequential(torch.nn.Conv3d(6, 1, 1), torch.nn.Softplus())
        common = dict(
            model=model,
            ct=ct,
            geometry=geometry,
            device=torch.device("cpu"),
            patch_size_zyx=(6, 6, 6),
            dose_scale=1.0,
            overlap=0.25,
            amp=False,
        )
        sequential = [
            predict_conditioned_array(condition=condition, batch_size=2, **common)
            for condition in conditions
        ]
        batched = predict_conditioned_arrays(
            conditions=conditions,
            condition_batch_size=2,
            **common,
        )
        np.testing.assert_allclose(
            np.stack(batched), np.stack(sequential), rtol=1.0e-6, atol=1.0e-6
        )


if __name__ == "__main__":
    unittest.main()
