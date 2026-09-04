from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from doserad_photon_ct import load_mha_array  # noqa: E402
from doserad_photon_ct.dataset_index import audit_dataset, expected_dose_name  # noqa: E402


MHA_HEADER = b"""ObjectType = Image
NDims = 3
BinaryData = True
ElementSpacing = 2 2 2
DimSize = 2 2 2
ElementType = MET_FLOAT
ElementDataFile = LOCAL
"""


class DatasetIndexTest(unittest.TestCase):
    def test_expected_dose_name(self) -> None:
        self.assertEqual(expected_dose_name(2, 7), "Dose_B2_CP007.mha")

    def test_complete_patient_is_added_to_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            patient_dir = root / "photon" / "training" / "1ABB001"
            (patient_dir / "image").mkdir(parents=True)
            (patient_dir / "dose").mkdir()
            (patient_dir / "image" / "ct.mha").write_bytes(MHA_HEADER)
            metadata = {
                "beams": [
                    {
                        "beam_idx": 0,
                        "SAD": 1000,
                        "iso_center": [0, 0, 0],
                        "num_mlc_leaf_pairs": 2,
                        "control_points": [
                            {
                                "cp_idx": 0,
                                "gantry_angle": 0,
                                "mlc_left_int_mm": [0, 1],
                                "mlc_right_int_mm": [2, 3],
                            }
                        ],
                    }
                ]
            }
            (patient_dir / "1ABB001.json").write_text(json.dumps(metadata), encoding="utf-8")
            (patient_dir / "dose" / "Dose_B0_CP000.mha").write_bytes(MHA_HEADER)

            summary = audit_dataset(root, check_headers=True)

            self.assertEqual(summary.patient_count, 1)
            self.assertEqual(summary.complete_patient_count, 1)
            self.assertTrue(summary.patients[0].complete_for_photon_ct)

    def test_load_compressed_mha_array(self) -> None:
        values = np.arange(8, dtype="<f4").reshape(2, 2, 2)
        header = b"""ObjectType = Image
NDims = 3
BinaryData = True
CompressedData = True
ElementSpacing = 2 2 2
DimSize = 2 2 2
ElementType = MET_FLOAT
ElementDataFile = LOCAL
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "image.mha"
            path.write_bytes(header + zlib.compress(values.tobytes()))
            actual = load_mha_array(path)
        np.testing.assert_array_equal(actual, values)


if __name__ == "__main__":
    unittest.main()
