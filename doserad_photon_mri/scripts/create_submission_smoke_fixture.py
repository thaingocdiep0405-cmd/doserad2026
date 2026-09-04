#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def write_mha(path: Path, array: np.ndarray) -> None:
    """Write an uncompressed 3D MetaImage.

    The data block is deliberately uncompressed: MetaIO needs
    ``CompressedDataSize`` to inflate a zlib block, and a header without it makes
    the reader emit "Uncompress failed" and hand back uninitialised memory
    instead of raising, so the fixture silently fed garbage to the container.
    """
    values = np.ascontiguousarray(array, dtype="<f4")
    z, y, x = values.shape
    header = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 3",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = False",
            "TransformMatrix = 1 0 0 0 1 0 0 0 1",
            "Offset = -7.5 -7.5 -7.5",
            "ElementSpacing = 1 1 1",
            f"DimSize = {x} {y} {z}",
            "ElementType = MET_FLOAT",
            "ElementDataFile = LOCAL",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((header + "\n").encode("ascii") + values.tobytes(order="C"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    image_dir = root / "images" / "radiation-dose-calculation-source-mri-image-1"
    image = np.zeros((16, 16, 16), dtype=np.float32)
    # A constant-intensity phantom is degenerate for MRI: the robust foreground
    # bounds collapse onto a single value, so the body mask removes the entire
    # volume and the smoke test yields an all-zero stack no matter what the
    # model does. A graded phantom keeps the low percentile below the tissue
    # values, which is how the mask behaves on real MRI.
    core = np.arange(12, dtype=np.float32).reshape(12, 1, 1) * 25.0 + 120.0
    image[2:-2, 2:-2, 2:-2] = core
    image[6:10, 6:10, 6:10] += 60.0
    write_mha(image_dir / "mr.mha", image)

    left = [-19.5] * 80
    right = [20.5] * 80
    metadata = [
        {
            "image_file_idx": 0,
            "anatomical_region": "abdominal",
            # Hidden-test format omits bookkeeping indices (beam_idx/cp_idx);
        # the fixture mirrors that so the smoke test exercises the fallback.
        "beams": [
                {
                    "SAD": 1000.0,
                    "iso_center": [0.0, 0.0, 0.0],
                    "num_mlc_leaf_pairs": 80,
                    "control_points": [
                        {
                            "cp_uuid": "smoke-test-control-point",
                            "gantry_angle": 0.0,
                            "mlc_left_int_mm": left,
                            "mlc_right_int_mm": right,
                            "output_info": {
                                "output_file_idx": 0,
                                "idx_in_output": 0,
                                "minimum_cutoff": 0.02,
                            },
                        }
                    ],
                }
            ],
        }
    ]
    (root / "stacked-photon-beam-level-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote smoke fixture to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
