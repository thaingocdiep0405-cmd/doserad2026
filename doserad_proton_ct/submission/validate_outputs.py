#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/output")
    for slot in range(1, 11):
        directory = root / "images" / f"stacked-radiation-dose-map-{slot}"
        files = list(directory.glob("*.mha"))
        if len(files) != 1:
            raise ValueError(f"slot {slot}: expected one MHA, found {len(files)}")
        image = sitk.ReadImage(str(files[0]))
        if image.GetDimension() != 4 or image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError(f"slot {slot}: output is not scalar 4D")
        values = sitk.GetArrayFromImage(image)
        if not np.isfinite(values).all():
            raise ValueError(f"slot {slot}: NaN/Inf detected")
        nonzero = values[values != 0]
        if slot == 1 and nonzero.size and float(nonzero.min()) <= 0.02:
            raise ValueError("slot 1: value at/below minimum_cutoff detected")
        print(
            f"slot={slot} size={image.GetSize()} min={float(values.min()):.6g} "
            f"max={float(values.max()):.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
