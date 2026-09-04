#!/usr/bin/env python3
"""Train an MRI to CT translation network for proton range calculation.

The pencil-beam engine needs mass density, which MRI does not carry. Treating
the body as water works for abdominal patients (MAE 0.013-0.033 on validation
beamlets) but fails in the thorax (0.18-0.29), because bone and lung both
appear dark on MRI while their densities sit at opposite extremes — 1.8 and
0.3. Intensity thresholds therefore cannot separate them; spatial context can,
and every training patient ships a CT registered to its MRI.

The loss weights the tissues the proton range actually depends on: an error of
50 HU in bone shifts the Bragg peak further than the same error in soft tissue.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, RandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE / "doserad_photon_ct" / "src"))

from doserad_photon_ct.dataset import crop_with_padding  # noqa: E402
from doserad_photon_ct.mha import load_mha_array  # noqa: E402
from doserad_photon_ct.model import ModelConfig, PhotonDoseUNet3D  # noqa: E402
from doserad_proton.data import load_split_patients, read_manifest  # noqa: E402

HU_MIN, HU_MAX = -1024.0, 2000.0


def normalize_hu(hu: np.ndarray) -> np.ndarray:
    """Map Hounsfield units onto [0, 1].

    The shared network ends in a softplus because it was built to predict dose,
    which is never negative. A symmetric [-1, 1] encoding would put everything
    below 488 HU — air, lung and every soft tissue — outside the reachable
    output range, so the encoding is shifted to the positive side instead.
    """
    return ((np.clip(hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def denormalize_hu(x: np.ndarray | torch.Tensor):
    return x * (HU_MAX - HU_MIN) + HU_MIN


class PairedDataset(Dataset):
    def __init__(self, manifest: Path, splits: Path, split: str, patch: int, seed: int):
        patients = load_split_patients(splits, split)
        pairs: dict[str, tuple[Path, Path]] = {}
        for record in read_manifest(manifest):
            if record.patient_id in patients:
                pairs.setdefault(record.patient_id, (record.mr_path, record.ct_path))
        self.items = sorted(pairs.items())
        if not self.items:
            raise ValueError(f"no patients for split {split!r}")
        self.patch = int(patch)
        self.seed = int(seed)
        self.deterministic = split != "train"
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.items) * 40

    def _volumes(self, patient: str, mr_path: Path, ct_path: Path):
        if patient not in self._cache:
            if len(self._cache) > 2:
                self._cache.pop(next(iter(self._cache)))
            mr = np.asarray(load_mha_array(mr_path), dtype=np.float32)
            ct = np.asarray(load_mha_array(ct_path), dtype=np.float32)
            if mr.shape != ct.shape:
                raise ValueError(f"{patient}: MRI {mr.shape} != CT {ct.shape}")
            # A robust upper percentile keeps the scaling stable against the
            # bright outliers MRI sequences occasionally contain.
            scale = float(np.percentile(mr[mr > 0], 99.0)) if (mr > 0).any() else 1.0
            self._cache[patient] = (mr / max(scale, 1e-6), ct)
        return self._cache[patient]

    def __getitem__(self, index: int):
        patient, (mr_path, ct_path) = self.items[index % len(self.items)]
        mr, ct = self._volumes(patient, mr_path, ct_path)
        rng = random.Random(self.seed + index) if self.deterministic else random
        size = np.array([self.patch] * 3)
        # Bias sampling towards the body so patches are not mostly background.
        for _ in range(32):
            start = np.array([rng.randrange(max(1, dim)) for dim in mr.shape]) - size // 2
            centre = np.clip(start + size // 2, 0, np.array(mr.shape) - 1)
            if mr[tuple(centre)] > 0.05:
                break
        mr_patch = crop_with_padding(mr, start, tuple(size), pad_value=0.0)
        ct_patch = crop_with_padding(ct, start, tuple(size), pad_value=HU_MIN)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(mr_patch)[None]),
            "target": torch.from_numpy(np.ascontiguousarray(normalize_hu(ct_patch))[None]),
        }


def weighted_loss(prediction: torch.Tensor, target: torch.Tensor, bone_weight: float, lung_weight: float):
    hu = denormalize_hu(target)
    weight = torch.ones_like(target)
    weight = torch.where(hu > 200.0, torch.full_like(weight, bone_weight), weight)
    weight = torch.where(hu < -300.0, torch.full_like(weight, lung_weight), weight)
    return (F.l1_loss(prediction, target, reduction="none") * weight).mean()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "manifest.csv")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "artifacts" / "splits.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "synthetic_ct")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--val-steps", type=int, default=100)
    parser.add_argument("--patch", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--bone-weight", type=float, default=3.0)
    parser.add_argument("--lung-weight", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--init-from", type=Path, default=None,
                        help="Warm start the weights from a checkpoint; the optimizer restarts.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ModelConfig(in_channels=1, base_channels=args.base_channels, levels=args.levels,
                         blocks_per_level=2, dropout=0.0)
    model = PhotonDoseUNet3D(config).to(device)
    if args.init_from is not None:
        model.load_state_dict(torch.load(args.init_from, map_location=device, weights_only=False)["model"])
        print(f"warm started from {args.init_from}", flush=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    loaders = {}
    for split in ("train", "validation"):
        dataset = PairedDataset(args.manifest, args.splits, split, args.patch, args.seed)
        steps = args.steps_per_epoch if split == "train" else args.val_steps
        sampler = RandomSampler(dataset, replacement=True, num_samples=steps * args.batch_size,
                                generator=torch.Generator().manual_seed(args.seed))
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                    num_workers=args.num_workers, pin_memory=device.type == "cuda",
                                    persistent_workers=args.num_workers > 0)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        stats = {}
        for split, loader in loaders.items():
            training = split == "train"
            model.train(training)
            totals = {"loss": 0.0, "mae_hu": 0.0, "bone_hu": 0.0, "n": 0}
            with torch.set_grad_enabled(training):
                for batch in loader:
                    inputs = batch["input"].to(device, non_blocking=True)
                    targets = batch["target"].to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                        outputs = model(inputs)
                        if isinstance(outputs, list):
                            outputs = outputs[0]
                        loss = weighted_loss(outputs, targets, args.bone_weight, args.lung_weight)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    with torch.no_grad():
                        hu_error = (denormalize_hu(outputs.float()) - denormalize_hu(targets.float())).abs()
                        bone = denormalize_hu(targets.float()) > 200.0
                        totals["loss"] += float(loss) * inputs.shape[0]
                        totals["mae_hu"] += float(hu_error.mean()) * inputs.shape[0]
                        totals["bone_hu"] += float(hu_error[bone].mean() if bone.any() else 0.0) * inputs.shape[0]
                        totals["n"] += inputs.shape[0]
            stats[split] = {k: totals[k] / max(totals["n"], 1) for k in ("loss", "mae_hu", "bone_hu")}
        scheduler.step()
        payload = {"epoch": epoch, "seconds": round(time.perf_counter() - started, 1),
                   "lr": optimizer.param_groups[0]["lr"], **{f"{s}_{k}": round(v, 4)
                   for s, d in stats.items() for k, v in d.items()}}
        with (args.output_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
        print(json.dumps(payload), flush=True)
        state = {"model_config": config.__dict__ if hasattr(config, "__dict__") else dict(config._asdict()),
                 "model": model.state_dict(), "epoch": epoch,
                 "hu_range": [HU_MIN, HU_MAX]}
        torch.save(state, args.output_dir / "last.pt")
        if stats["validation"]["mae_hu"] < best:
            best = stats["validation"]["mae_hu"]
            torch.save(state, args.output_dir / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
