import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from doserad_photon_ct.losses import DoseLoss, LossConfig  # noqa: E402
from doserad_photon_ct.model import ModelConfig, PhotonDoseUNet3D  # noqa: E402


class ScaleHeadTest(unittest.TestCase):
    def test_scale_head_initializes_neutral_and_stays_bounded(self) -> None:
        torch.manual_seed(0)
        base = PhotonDoseUNet3D(
            ModelConfig(in_channels=6, base_channels=8, levels=2)
        ).eval()
        calibrated = PhotonDoseUNet3D(
            ModelConfig(in_channels=6, base_channels=8, levels=2, scale_head=True)
        ).eval()
        # Copy shared weights so only the head differs.
        state = {
            key: value
            for key, value in base.state_dict().items()
        }
        calibrated.load_state_dict(state, strict=False)
        x = torch.randn(2, 6, 16, 16, 16)
        with torch.no_grad():
            reference = base(x)
            output = calibrated(x)
        # Zero-initialized final layer -> exp(0) = 1 -> identical output.
        torch.testing.assert_close(output, reference)

        # Push the head hard: the correction must stay inside exp(+-0.5).
        with torch.no_grad():
            calibrated.scale_head[-1].bias.fill_(100.0)
            inflated = calibrated(x)
        ratio = inflated / reference.clamp_min(1.0e-8)
        self.assertLessEqual(float(ratio.max()), float(torch.exp(torch.tensor(0.5))) + 1e-4)

    def test_scale_head_is_trainable_end_to_end(self) -> None:
        torch.manual_seed(0)
        model = PhotonDoseUNet3D(
            ModelConfig(in_channels=6, base_channels=8, levels=2, scale_head=True)
        )
        x = torch.randn(1, 6, 16, 16, 16)
        out = model(x)
        out.mean().backward()
        grads = [p.grad for p in model.scale_head.parameters()]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))


class OutOfFieldLossTest(unittest.TestCase):
    def test_penalizes_dose_in_cold_regions_only(self) -> None:
        target = torch.zeros(1, 1, 8, 8, 8)
        target[0, 0, :, :4] = 1.0  # hot half
        maximum = torch.tensor([1.0])
        criterion = DoseLoss(LossConfig(out_of_field_weight=1.0))

        clean = torch.where(target > 0, torch.full_like(target, 0.9), torch.zeros_like(target))
        leaky = torch.where(target > 0, torch.full_like(target, 0.9), torch.full_like(target, 0.2))
        _, clean_parts = criterion(clean, target, maximum)
        _, leaky_parts = criterion(leaky, target, maximum)
        self.assertAlmostEqual(float(clean_parts["out_of_field"]), 0.0, places=6)
        self.assertAlmostEqual(float(leaky_parts["out_of_field"]), 0.2, places=6)



class PBDistillationTest(unittest.TestCase):
    def test_pb_head_returns_tuple_in_training_only(self) -> None:
        torch.manual_seed(0)
        model = PhotonDoseUNet3D(
            ModelConfig(in_channels=6, base_channels=8, levels=2, pb_head=True)
        )
        x = torch.randn(1, 6, 16, 16, 16)
        model.train()
        out = model(x)
        self.assertIsInstance(out, tuple)
        main, pb = out
        self.assertEqual(main.shape, pb.shape)
        model.eval()
        with torch.no_grad():
            out = model(x)
        self.assertIsInstance(out, torch.Tensor)

    def test_pb_loss_only_counts_valid_samples(self) -> None:
        criterion = DoseLoss(LossConfig(pb_distill_weight=1.0))
        prediction = torch.rand(2, 1, 8, 8, 8)
        target = torch.rand(2, 1, 8, 8, 8)
        pb_prediction = torch.rand(2, 1, 8, 8, 8)
        pb_target = pb_prediction.clone()
        pb_target[1] += 10.0  # only the invalid sample disagrees
        valid = torch.tensor([True, False])
        _, parts = criterion(
            prediction,
            target,
            pb_prediction=pb_prediction,
            pb_target=pb_target,
            pb_valid=valid,
        )
        self.assertIn("pb_distill", parts)
        self.assertAlmostEqual(float(parts["pb_distill"]), 0.0, places=6)
        _, parts_all = criterion(
            prediction,
            target,
            pb_prediction=pb_prediction,
            pb_target=pb_target,
            pb_valid=torch.tensor([True, True]),
        )
        self.assertGreater(float(parts_all["pb_distill"]), 1.0)

if __name__ == "__main__":
    unittest.main()
