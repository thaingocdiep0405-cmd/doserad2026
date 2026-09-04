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
    parser = argparse.ArgumentParser(description="Create Proton CT/MRI submission fixture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("proton-ct", "proton-mri"), required=True)
    parser.add_argument(
        "--beamlets",
        type=int,
        default=1,
        help="beamlets in output slot 1; >1 exercises the multi-frame stack path",
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=31.729,
        # Lowest energy in the machine table. Its range in water is about 1 cm,
        # so the Bragg peak falls inside the 16 mm fixture box and the smoke
        # test yields non-zero dose instead of an all-zero stack.
        help="beamlet energy in MeV; must exist in the machine table",
    )
    args = parser.parse_args()
    if args.beamlets < 1:
        raise SystemExit("--beamlets must be positive")

    modality = "ct" if args.task == "proton-ct" else "mri"
    root = args.output
    image_dir = (
        root / "images" / f"radiation-dose-calculation-source-{modality}-image-1"
    )
    image = np.zeros((16, 16, 16), dtype=np.float32)
    if modality == "ct":
        image[[0, -1], :, :] = -1024.0
        image[:, [0, -1], :] = -1024.0
        image[:, :, [0, -1]] = -1024.0
    else:
        # A constant-intensity phantom is degenerate for MRI: the robust
        # foreground bounds collapse onto a single value, so the body mask
        # (image <= bounds[0]) removes the entire volume and the smoke test
        # yields an all-zero stack no matter what the model does. A graded
        # phantom keeps the 0.5th percentile below the tissue values, which is
        # how the mask behaves on real MRI.
        core = np.arange(12, dtype=np.float32).reshape(12, 1, 1) * 25.0 + 120.0
        image[2:-2, 2:-2, 2:-2] = core
        image[6:10, 6:10, 6:10] += 60.0
    write_mha(image_dir / f"{modality}.mha", image)

    # Descending idx_in_output so a run that ignores output_info and relies on
    # arrival order produces a visibly wrong stack.
    beamlets = [
        {
            "beamlet_idx": order,
            "beamlet_uuid": f"smoke-test-beamlet-{order}",
            "energy": args.energy,
            "output_info": {
                "output_file_idx": 0,
                "idx_in_output": args.beamlets - 1 - order,
                "minimum_cutoff": 0.02,
            },
        }
        for order in range(args.beamlets)
    ]

    metadata = [
        {
            "image_file_idx": 0,
            "anatomical_region": "abdominal",
            "iso_center": [0.0, 0.0, 0.0],
            "beams": [
                {
                    "beam_idx": 0,
                    "gantry_angle": 0.0,
                    "rays": [
                        {
                            "ray_idx": 0,
                            "ray_source": [-500.0, 0.0, 0.0],
                            "ray_target": [0.0, 0.0, 0.0],
                            "beamlets": beamlets,
                        }
                    ],
                }
            ],
        }
    ]
    (root / "stacked-proton-beam-level-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {args.task} smoke fixture to {root} "
        f"({args.beamlets} beamlet{'s' if args.beamlets != 1 else ''})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
