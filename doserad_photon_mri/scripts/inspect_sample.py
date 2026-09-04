#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from doserad_photon_mri import read_mha_header  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one photon-CT manifest sample.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "manifest.csv",
    )
    parser.add_argument("--row", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("manifest has no rows")
    if not 0 <= args.row < len(rows):
        raise IndexError(f"row must be between 0 and {len(rows) - 1}")

    row = rows[args.row]
    ct_header = read_mha_header(Path(row["ct_path"]))
    dose_header = read_mha_header(Path(row["dose_path"]))
    output = {
        "sample": row,
        "ct_geometry": {
            key: ct_header.get(key)
            for key in ("NDims", "DimSize", "ElementSpacing", "Offset", "ElementType")
        },
        "dose_geometry": {
            key: dose_header.get(key)
            for key in ("NDims", "DimSize", "ElementSpacing", "Offset", "ElementType")
        },
        "same_grid": all(
            ct_header.get(key) == dose_header.get(key)
            for key in ("NDims", "DimSize", "ElementSpacing", "Offset", "TransformMatrix")
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
