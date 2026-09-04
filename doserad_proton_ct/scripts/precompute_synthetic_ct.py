#!/usr/bin/env python3
"""Write a synthetic CT beside every patient's MRI.

The dose network needs mass density, and computing it inside the data loader
would put a second network on the GPU that the trainer is already saturating.
The translation depends only on the MRI, so it is done once here and read back
as an image like any other.

Training uses these synthetic volumes rather than the real CTs the dataset also
ships: inference only ever sees a synthetic one, and a network trained on real
density would meet a distribution at test time that it never saw.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE / "doserad_photon_ct" / "src"))

from doserad_photon_ct.model import ModelConfig, PhotonDoseUNet3D  # noqa: E402
from doserad_proton.data import read_manifest  # noqa: E402

HU_MIN, HU_MAX = -1024.0, 2000.0


@torch.no_grad()
def synthesize(model, mri: np.ndarray, device, patch: int, stride: int) -> np.ndarray:
    scale = float(np.percentile(mri[mri > 0], 99.0)) if (mri > 0).any() else 1.0
    scaled = mri / max(scale, 1e-6)
    total = np.zeros(mri.shape, np.float32)
    count = np.zeros(mri.shape, np.float32)
    starts = []
    for axis, extent in enumerate(mri.shape):
        axis_starts = list(range(0, max(extent - patch, 0) + 1, stride)) or [0]
        if axis_starts[-1] + patch < extent:
            axis_starts.append(max(extent - patch, 0))
        starts.append(axis_starts)
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                window = (slice(z, z + patch), slice(y, y + patch), slice(x, x + patch))
                block = scaled[window]
                padding = [(0, patch - size) for size in block.shape]
                padded = np.pad(block, padding) if any(p[1] for p in padding) else block
                batch = torch.from_numpy(np.ascontiguousarray(padded))[None, None].to(device)
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    predicted = model(batch)
                patch_out = predicted.float().cpu().numpy()[0, 0]
                total[window] += patch_out[: block.shape[0], : block.shape[1], : block.shape[2]]
                count[window] += 1.0
    return ((total / np.maximum(count, 1e-6)) * (HU_MAX - HU_MIN) + HU_MIN).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "manifest.csv")
    parser.add_argument("--name", default="sct.mha", help="File name written into each image directory.")
    parser.add_argument("--patch", type=int, default=96)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = PhotonDoseUNet3D(ModelConfig(**config) if isinstance(config, dict) else config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    print(f"sCT tu {args.checkpoint} (epoch {checkpoint.get('epoch')})", flush=True)

    mri_paths: dict[str, Path] = {}
    for record in read_manifest(args.manifest):
        mri_paths.setdefault(record.patient_id, record.mr_path)

    started = time.perf_counter()
    for index, (patient, mri_path) in enumerate(sorted(mri_paths.items()), start=1):
        destination = mri_path.parent / args.name
        if destination.exists() and not args.overwrite:
            print(f"[{index}/{len(mri_paths)}] {patient}: da co, bo qua", flush=True)
            continue
        reference = sitk.ReadImage(str(mri_path))
        mri = sitk.GetArrayFromImage(reference).astype(np.float32)
        hu = synthesize(model, mri, device, args.patch, args.stride)
        # HU is integral in the source CTs, so int16 is lossless here and keeps
        # 75 volumes at a size the disk will not notice.
        image = sitk.GetImageFromArray(np.clip(hu, HU_MIN, HU_MAX).astype(np.int16))
        image.CopyInformation(reference)
        sitk.WriteImage(image, str(destination), useCompression=True)
        elapsed = time.perf_counter() - started
        print(f"[{index}/{len(mri_paths)}] {patient}: {destination.name} "
              f"({destination.stat().st_size / 1e6:.1f} MB, {elapsed / index:.1f}s/BN)", flush=True)
    print("xong", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
