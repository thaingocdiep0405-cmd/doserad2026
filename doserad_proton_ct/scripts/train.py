#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, RandomSampler, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE / "doserad_photon_ct" / "src"))

from doserad_photon_ct.losses import DoseLoss, LossConfig, masked_beam_mae_tensor, normalized_rmse_tensor  # noqa: E402
from doserad_photon_ct.model import ModelConfig, PhotonDoseUNet3D  # noqa: E402
from doserad_proton.data import ProtonPatchDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DoseRAD2026 proton CT/MRI model")
    parser.add_argument("--modality", choices=("ct", "mri"), required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "manifest.csv")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "artifacts" / "splits.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--val-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--dose-scale", type=float, required=True)
    parser.add_argument("--positive-patch-probability", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=12)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--blocks-per-level", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--high-dose-weight", type=float, default=3.0)
    parser.add_argument("--official-mae-weight", type=float, default=2.0)
    parser.add_argument("--gradient-loss-weight", type=float, default=0.1)
    parser.add_argument("--idd-surrogate-weight", type=float, default=0.1)
    parser.add_argument("--scale-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--scale-head",
        action="store_true",
        help="Predict a bounded per-beamlet global dose-scale correction",
    )
    parser.add_argument("--out-of-field-weight", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--deep-supervision", action="store_true")
    parser.add_argument("--deep-supervision-weight", type=float, default=0.3)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug-energy-jitter", type=float, default=0.02)
    parser.add_argument("--aug-density-scale", type=float, default=0.03)
    parser.add_argument("--aug-noise-std", type=float, default=0.02)
    parser.add_argument("--ray-gate-threshold", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop after this many epochs without validation improvement; 0 disables it")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--range-channels", action="store_true",
                        help="add WEPL + analytic Bragg-peak prior channels (12 input channels)")
    parser.add_argument("--synthetic-ct", default=None,
                        help="File name of a synthetic CT written beside each MRI "
                             "(see scripts/precompute_synthetic_ct.py). Supplies the "
                             "density channel and the water-equivalent depth, which "
                             "MRI cannot provide on its own.")
    parser.add_argument("--snapshot-every", type=int, default=0,
                        help="also save epochNNN.pt every N epochs for later full-volume selection")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


class Meter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def add(self, value: float, count: int):
        if math.isfinite(value):
            self.total += value * count
            self.count += count

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")


def make_loader(args: argparse.Namespace, split: str):
    aug_kwargs = {}
    if hasattr(args, 'augment') and args.augment and split == "train":
        aug_kwargs = {
            "augment": True,
            "aug_energy_jitter": args.aug_energy_jitter,
            "aug_density_scale": args.aug_density_scale,
            "aug_noise_std": args.aug_noise_std,
        }
    dataset = ProtonPatchDataset(
        args.manifest, args.splits, split, modality=args.modality,
        patch_size_zyx=tuple(args.patch_size), dose_scale=args.dose_scale,
        positive_patch_probability=args.positive_patch_probability,
        cache_size=args.cache_size, seed=args.seed,
        deterministic_sampling=split == "validation",
        include_range_channels=args.range_channels,
        synthetic_ct_name=args.synthetic_ct,
        **aug_kwargs,
    )
    samples = (args.steps_per_epoch if split == "train" else args.val_steps) * args.batch_size
    if split == "train":
        sampler = RandomSampler(dataset, replacement=True, num_samples=samples,
                                generator=torch.Generator().manual_seed(args.seed))
        selected = dataset
    else:
        count = min(len(dataset), samples)
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed + 1))[:count]
        selected = Subset(dataset, indices.tolist())
        sampler = None
    loader = DataLoader(
        selected, batch_size=args.batch_size, sampler=sampler,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.device in ("cuda", "auto") and torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    return dataset, loader


def run_epoch(model, loader, criterion, device, args, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    meters: dict[str, Meter] = defaultdict(Meter)
    started = time.perf_counter()
    if training:
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad if training else torch.no_grad
    grad_accum = getattr(args, 'gradient_accumulation', 1)
    with context():
        for step, batch in enumerate(loader, 1):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                                enabled=not args.no_amp):
                predictions = model(inputs)
                main_pred = predictions[0] if isinstance(predictions, list) else predictions
                if args.ray_gate_threshold > 0.0:
                    if isinstance(predictions, list):
                        gate = inputs[:, 2:3] >= args.ray_gate_threshold
                        predictions = [p * gate for p in predictions]
                        main_pred = predictions[0]
                    else:
                        predictions = predictions * (inputs[:, 2:3] >= args.ray_gate_threshold)
                        main_pred = predictions
                loss, components = criterion(predictions, targets, batch["target_max"],
                                             inputs[:, 1:2] > 0.5, batch["gantry_angle_deg"])
            if training:
                backward_loss = loss / grad_accum
                scaler.scale(backward_loss).backward()
                if step % grad_accum == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            count = inputs.shape[0]
            meters["loss"].add(float(loss.detach()), count)
            meters["masked_mae"].add(float(masked_beam_mae_tensor(main_pred, targets, batch["target_max"])), count)
            meters["nrmse"].add(float(normalized_rmse_tensor(main_pred, targets, batch["target_max"])), count)
            for name, value in components.items():
                meters[f"loss_{name}"].add(float(value.detach()), count)
            if training and (step % args.log_every == 0 or step == len(loader)):
                rate = step * args.batch_size / max(time.perf_counter() - started, 1e-6)
                print(f"step={step}/{len(loader)} loss={meters['loss'].mean:.6f} mae={meters['masked_mae'].mean:.6f} samples/s={rate:.2f}", flush=True)
    return {name: meter.mean for name, meter in meters.items()}


def save(path: Path, model, optimizer, scheduler, scaler, epoch, best, config, args):
    payload = {
        "format_version": 1, "task": f"proton_{args.modality}", "epoch": epoch,
        "best_metric": best, "model_config": config.to_dict(), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "training_config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("resume and init-checkpoint are mutually exclusive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed); torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, train_loader = make_loader(args, "train")
    _, val_loader = make_loader(args, "validation")
    config = ModelConfig(in_channels=12 if args.range_channels else 10,
                         base_channels=args.base_channels,
                         levels=args.levels, blocks_per_level=args.blocks_per_level,
                         dropout=args.dropout, physics_priors=True,
                         scale_head=args.scale_head,
                         deep_supervision=args.deep_supervision)
    model = PhotonDoseUNet3D(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.no_amp)
    criterion = DoseLoss(LossConfig(
        high_dose_weight=args.high_dose_weight, gradient_weight=args.gradient_loss_weight,
        official_mae_weight=args.official_mae_weight, idd_surrogate_weight=args.idd_surrogate_weight,
        scale_weight=args.scale_loss_weight,
        out_of_field_weight=args.out_of_field_weight,
        deep_supervision_weight=args.deep_supervision_weight if args.deep_supervision else 0.0,
    ))
    start_epoch, best, epochs_without_improvement = 1, float("inf"), 0
    initial_path = args.resume or args.init_checkpoint
    if initial_path:
        checkpoint = torch.load(initial_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if unexpected:
            raise ValueError(f"init checkpoint has unexpected keys: {unexpected[:5]}")
        if missing:
            if not all(key.startswith("scale_head.") for key in missing):
                raise ValueError(f"init checkpoint is missing keys: {missing[:5]}")
            print(f"Initialized {len(missing)} new scale-head parameters from scratch")
        if args.resume:
            optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint.get("scaler", {})); start_epoch = int(checkpoint["epoch"]) + 1
            best = float(checkpoint.get("best_metric", best))
        print(f"Loaded {initial_path}", flush=True)
    config_payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config_payload["model"] = config.to_dict()
    (args.output_dir / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n")
    print(f"Task=proton-{args.modality} device={device} train={len(train_dataset)} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        started = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, device, args, optimizer, scaler)
        validation_metrics = run_epoch(model, val_loader, criterion, device, args)
        scheduler.step()
        metric = validation_metrics["masked_mae"]
        improved = math.isfinite(metric) and metric < best - args.early_stopping_min_delta
        if improved:
            best = metric
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        payload = {"epoch": epoch, "seconds": time.perf_counter() - started,
                   "learning_rate": optimizer.param_groups[0]["lr"], "train": train_metrics,
                   "validation": validation_metrics, "best_metric": best}
        with (args.output_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(payload, allow_nan=True) + "\n")
        save(args.output_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best, config, args)
        if improved:
            save(args.output_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best, config, args)
        # Patch validation ranks checkpoints differently from the full-volume
        # metrics the leaderboard scores, so periodic snapshots keep the real
        # candidates available for a full-volume comparison after training.
        if args.snapshot_every > 0 and epoch % args.snapshot_every == 0:
            save(
                args.output_dir / f"epoch{epoch:03d}.pt",
                model, optimizer, scheduler, scaler, epoch, best, config, args,
            )
        print(json.dumps(payload, allow_nan=True), flush=True)
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping after {epochs_without_improvement} epochs without validation improvement",
                flush=True,
            )
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
