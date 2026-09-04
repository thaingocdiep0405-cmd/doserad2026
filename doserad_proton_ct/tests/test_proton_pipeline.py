from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(WORKSPACE / "doserad_photon_ct" / "src"))

from doserad_photon_ct.conditioning import SpatialGeometry
from doserad_proton.conditioning import ProtonCondition, build_proton_channels
from doserad_proton.inference import predict_conditioned_arrays


def test_ct_and_mri_share_ten_channel_contract():
    geometry = SpatialGeometry((2.0, 2.0, 2.0), (-16.0, -16.0, -16.0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    condition = ProtonCondition(0.0, (0.0, -1000.0, 0.0), (0.0, 0.0, 0.0), 150.3508, 1.0845, 4.7472)
    ct = np.zeros((16, 16, 16), dtype=np.float32)
    mri = np.ones_like(ct)
    ct_channels = build_proton_channels(ct, (0, 0, 0), geometry, condition, modality="ct")
    mr_channels = build_proton_channels(mri, (0, 0, 0), geometry, condition, modality="mri", intensity_bounds=(0.0, 2.0))
    assert ct_channels.shape == mr_channels.shape == (10, 16, 16, 16)
    assert np.isfinite(ct_channels).all()
    assert np.isfinite(mr_channels).all()
    assert float(ct_channels[2].max()) <= 1.0


class _FluenceModel(torch.nn.Module):
    def forward(self, inputs):
        return inputs[:, 2:3]


def test_batched_inference_preserves_condition_order_and_ray_gate():
    geometry = SpatialGeometry((2.0, 2.0, 2.0), (-16.0, -16.0, -16.0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    conditions = [
        ProtonCondition(0.0, (0.0, -1000.0, 0.0), (0.0, 0.0, 0.0), 100.0, 2.0, 4.0),
        ProtonCondition(0.0, (8.0, -1000.0, 0.0), (8.0, 0.0, 0.0), 150.0, 1.0, 6.0),
    ]
    image = np.zeros((16, 16, 16), dtype=np.float32)
    predictions = predict_conditioned_arrays(
        _FluenceModel(), image=image, geometry=geometry, conditions=conditions,
        modality="ct", device=torch.device("cpu"), patch_size_zyx=(16, 16, 16),
        dose_scale=2.0, overlap=0.0, condition_batch_size=2, amp=False,
        skip_empty_ray=False, mask_outside_body=False, relative_cutoff=0.0,
        ray_gate_threshold=0.2,
    )
    assert len(predictions) == 2
    for prediction, condition in zip(predictions, conditions):
        fluence = build_proton_channels(image, (0, 0, 0), geometry, condition, modality="ct")[2]
        expected = 2.0 * fluence * (fluence >= 0.2)
        np.testing.assert_allclose(prediction, expected, rtol=1e-6, atol=1e-6)
    assert not np.array_equal(predictions[0], predictions[1])
