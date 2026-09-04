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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from doserad_photon_ct.dataset import PhotonCTPatchDataset, lookup_condition  # noqa: E402
from doserad_photon_ct.dataset_index import read_mha_header  # noqa: E402
from doserad_photon_ct.inference import predict_record_volume  # noqa: E402
from doserad_photon_ct.losses import (  # noqa: E402
    DoseLoss,
    LossConfig,
    masked_beam_mae_tensor,
    normalized_rmse_tensor,
)
from doserad_photon_ct.mha import load_mha_array  # noqa: E402
from doserad_photon_ct.model import ModelConfig, PhotonDoseUNet3D  # noqa: E402
from doserad_photon_ct.metrics import beam_direction, idd_curve_distance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DoseRAD2026 photon-CT patch baseline.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts/manifest.csv")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "artifacts/splits.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/photon_ct_baseline")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Warm-start model weights only; supports expanding 6 input channels to 7",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--val-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ct-cache-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, nargs=3, metavar=("Z", "Y", "X"), default=(96, 96, 96))
    parser.add_argument("--dose-scale", type=float, default=1.0e-4)
    parser.add_argument("--ct-clip", type=float, nargs=2, metavar=("LOW", "HIGH"), default=(-1024.0, 2000.0))
    parser.add_argument("--positive-patch-probability", type=float, default=0.8)
    parser.add_argument("--include-density", action="store_true")
    parser.add_argument(
        "--physics-priors",
        action="store_true",
        help="Add density, aperture-edge, inverse-square, field-width and fluence priors",
    )
    parser.add_argument(
        "--radiological-depth",
        action="store_true",
        help="Add ray-traced water-equivalent depth and attenuated-fluence "
        "channels (implies --physics-priors)",
    )

    parser.add_argument("--base-channels", type=int, default=12)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--blocks-per-level", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--high-dose-weight", type=float, default=4.0)
    parser.add_argument("--gradient-loss-weight", type=float, default=0.1)
    parser.add_argument("--official-mae-weight", type=float, default=0.0)
    parser.add_argument("--idd-surrogate-weight", type=float, default=0.0)
    parser.add_argument("--scale-loss-weight", type=float, default=0.0)
    parser.add_argument("--out-of-field-weight", type=float, default=0.0)
    parser.add_argument("--out-of-field-threshold", type=float, default=0.02)
    parser.add_argument(
        "--pb-dose-dir",
        type=Path,
        help="Directory of precomputed pencil-beam doses "
        "(<pid>_B<b>_CP<ccc>.npz) for auxiliary distillation",
    )
    parser.add_argument("--pb-distill-weight", type=float, default=0.0)
    parser.add_argument(
        "--scale-head",
        action="store_true",
        help="Predict a bounded per-sample global dose-scale correction",
    )
    parser.add_argument("--deep-supervision", action="store_true")
    parser.add_argument("--deep-supervision-weight", type=float, default=0.3)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug-hu-jitter", type=float, default=50.0)
    parser.add_argument("--aug-density-scale", type=float, default=0.05)
    parser.add_argument("--aug-noise-std", type=float, default=0.02)

    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument("--full-val-every", type=int, default=0)
    parser.add_argument("--full-val-samples", type=int, default=1)
    parser.add_argument(
        "--full-val-records-per-patient",
        type=int,
        default=1,
        help="Full-volume CPs per selected patient, balanced across beams and arc",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("patch_mae", "full_mae"),
        default="full_mae",
        help="Metric used for best.pt; full_mae requires periodic full validation",
    )
    parser.add_argument("--inference-overlap", type=float, default=0.5)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--inference-patch-size", type=int, nargs=3)
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        if math.isfinite(value):
            self.total += value * count
            self.count += count

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else float("nan")


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    # Each worker computing radiological depth with the default intra-op
    # pool (one thread per core) oversubscribes the CPU catastrophically
    # once anything else runs on the machine; one thread per worker is
    # plenty (~100 ms per item) and keeps throughput stable.
    torch.set_num_threads(1)


def make_loaders(
    args: argparse.Namespace,
) -> tuple[PhotonCTPatchDataset, DataLoader, PhotonCTPatchDataset, DataLoader]:
    common: dict[str, Any] = {
        "manifest_path": args.manifest,
        "splits_path": args.splits,
        "patch_size_zyx": tuple(args.patch_size),
        "dose_scale": args.dose_scale,
        "ct_clip": tuple(args.ct_clip),
        "positive_patch_probability": args.positive_patch_probability,
        "ct_cache_size": args.ct_cache_size,
        "seed": args.seed,
        "include_density": args.include_density,
        "include_physics_priors": args.physics_priors,
        "include_radiological_depth": args.radiological_depth,
        "pb_dose_dir": args.pb_dose_dir,
    }
    train_dataset = PhotonCTPatchDataset(
        split="train",
        deterministic_sampling=False,
        augment=args.augment,
        aug_hu_jitter=args.aug_hu_jitter,
        aug_density_scale=args.aug_density_scale,
        aug_noise_std=args.aug_noise_std,
        **common,
    )
    validation_dataset = PhotonCTPatchDataset(
        split="validation", deterministic_sampling=True, **common
    )

    train_samples = args.steps_per_epoch * args.batch_size
    train_sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=train_samples,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_count = min(len(validation_dataset), args.val_steps * args.batch_size)
    validation_indices = torch.randperm(
        len(validation_dataset), generator=torch.Generator().manual_seed(args.seed + 1)
    )[:validation_count].tolist()
    validation_subset = Subset(validation_dataset, validation_indices)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": worker_init_fn,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    validation_loader = DataLoader(validation_subset, shuffle=False, **loader_kwargs)
    return train_dataset, train_loader, validation_dataset, validation_loader


def autocast_context(device: torch.device, enabled: bool):
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DoseLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    meters: dict[str, AverageMeter] = defaultdict(AverageMeter)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    amp_enabled = not args.no_amp and device.type == "cuda"

    for step, batch in enumerate(loader, start=1):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            predictions = model(inputs)
            pb_prediction = None
            if isinstance(predictions, tuple):
                predictions, pb_prediction = predictions
            loss, components = criterion(
                predictions,
                targets,
                batch.get("target_max"),
                inputs[:, 1:2] > 0.5,
                batch.get("gantry_angle_deg"),
                pb_prediction=pb_prediction,
                pb_target=(
                    batch["pb_target"].to(device, non_blocking=True)
                    if pb_prediction is not None and "pb_target" in batch
                    else None
                ),
                pb_valid=batch.get("has_pb"),
            )
            backward_loss = loss / args.gradient_accumulation
        scaler.scale(backward_loss).backward()

        if step % args.gradient_accumulation == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            meters["grad_norm"].update(float(grad_norm))

        main_pred = predictions[0] if isinstance(predictions, list) else predictions
        batch_size = inputs.shape[0]
        meters["loss"].update(float(loss.detach()), batch_size)
        for name, value in components.items():
            meters[f"loss_{name}"].update(float(value.detach()), batch_size)
        meters["masked_mae"].update(
            float(masked_beam_mae_tensor(main_pred, targets, batch.get("target_max"))),
            batch_size,
        )
        meters["nrmse"].update(
            float(normalized_rmse_tensor(main_pred, targets, batch.get("target_max"))),
            batch_size,
        )

        if step % args.log_every == 0 or step == len(loader):
            elapsed = time.perf_counter() - started
            print(
                f"epoch={epoch} step={step}/{len(loader)} "
                f"loss={meters['loss'].average:.6f} "
                f"masked_mae={meters['masked_mae'].average:.5f} "
                f"samples/s={(step * args.batch_size) / max(elapsed, 1e-6):.2f}",
                flush=True,
            )
    return {name: meter.average for name, meter in meters.items()}


@torch.no_grad()
def validate_patches(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DoseLoss,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    meters: dict[str, AverageMeter] = defaultdict(AverageMeter)
    amp_enabled = not args.no_amp and device.type == "cuda"
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            predictions = model(inputs)
            loss, components = criterion(
                predictions,
                targets,
                batch.get("target_max"),
                inputs[:, 1:2] > 0.5,
                batch.get("gantry_angle_deg"),
            )
        main_pred = predictions[0] if isinstance(predictions, list) else predictions
        batch_size = inputs.shape[0]
        meters["loss"].update(float(loss), batch_size)
        for name, value in components.items():
            meters[f"loss_{name}"].update(float(value), batch_size)
        meters["masked_mae"].update(
            float(masked_beam_mae_tensor(main_pred, targets, batch.get("target_max"))),
            batch_size,
        )
        meters["nrmse"].update(
            float(normalized_rmse_tensor(main_pred, targets, batch.get("target_max"))),
            batch_size,
        )
    return {name: meter.average for name, meter in meters.items()}


@torch.no_grad()
def validate_full_volumes(
    model: torch.nn.Module,
    dataset: PhotonCTPatchDataset,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    # Pick at most one deterministic control point per patient first. The old
    # implementation selected the first N rows, which all belonged to one
    # patient and badly underestimated validation variance.
    records_by_patient: dict[str, list] = defaultdict(list)
    for record in dataset.records:
        records_by_patient[record.patient_id].append(record)
    rng = random.Random(args.seed + 42_424)
    patient_ids = sorted(records_by_patient)
    rng.shuffle(patient_ids)
    records = []
    records_per_patient = max(1, args.full_val_records_per_patient)
    for patient_id in patient_ids[: max(0, args.full_val_samples)]:
        by_beam: dict[int, list] = defaultdict(list)
        for record in records_by_patient[patient_id]:
            by_beam[record.beam_idx].append(record)
        beam_ids = sorted(by_beam)
        for slot in range(records_per_patient):
            beam_records = sorted(
                by_beam[beam_ids[slot % len(beam_ids)]], key=lambda item: item.cp_idx
            )
            round_index = slot // len(beam_ids)
            rounds = math.ceil(records_per_patient / len(beam_ids))
            position = min(
                len(beam_records) - 1,
                int((round_index + 0.5) * len(beam_records) / rounds),
            )
            records.append(beam_records[position])
    per_patient: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"masked_mae": [], "nrmse": [], "idd_distance": []}
    )
    for record in records:
        print(
            f"full validation: {record.patient_id} B{record.beam_idx} CP{record.cp_idx:03d}",
            flush=True,
        )
        prediction = predict_record_volume(
            model,
            record,
            device=device,
            patch_size_zyx=tuple(args.inference_patch_size or args.patch_size),
            dose_scale=args.dose_scale,
            ct_clip=tuple(args.ct_clip),
            overlap=args.inference_overlap,
            batch_size=args.inference_batch_size,
            amp=not args.no_amp,
            include_density=args.include_density,
            include_physics_priors=args.physics_priors,
            include_radiological_depth=args.radiological_depth,
            skip_empty_aperture=True,
            mask_outside_body=True,
        )
        target = np.asarray(load_mha_array(record.dose_path), dtype=np.float32)
        target_max = float(target.max())
        if target_max <= 0:
            continue
        mask = target >= 0.1 * target_max
        patient_values = per_patient[record.patient_id]
        patient_values["masked_mae"].append(
            float(np.mean(np.abs(prediction[mask] - target[mask])) / target_max)
        )
        patient_values["nrmse"].append(
            float(np.sqrt(np.mean((prediction - target) ** 2)) / target_max)
        )
        spacing_xyz = tuple(
            float(value)
            for value in read_mha_header(record.ct_path)["ElementSpacing"].split()
        )
        condition = lookup_condition(record)
        patient_values["idd_distance"].append(
            idd_curve_distance(
                prediction,
                target,
                beam_direction(condition.gantry_angle_deg),
                spacing_xyz,
            )
        )
    result = {}
    for key in ("masked_mae", "nrmse", "idd_distance"):
        patient_means = [
            float(np.nanmean(values[key]))
            for values in per_patient.values()
            if values[key]
        ]
        result[key] = (
            float(np.nanmean(patient_means)) if patient_means else float("nan")
        )
    result["patients"] = float(len(per_patient))
    result["records"] = float(len(records))
    return result


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_metric: float,
    best_patch_metric: float,
    best_full_metric: float,
    best_idd_metric: float,
    model_config: ModelConfig,
    args: argparse.Namespace,
) -> None:
    unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
    payload = {
        "format_version": 1,
        "epoch": epoch,
        "best_metric": best_metric,
        "best_patch_metric": best_patch_metric,
        "best_full_metric": best_full_metric,
        "best_idd_metric": best_idd_metric,
        "model_config": model_config.to_dict(),
        "training_config": vars(args),
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def append_metrics(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, allow_nan=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.steps_per_epoch <= 0 or args.val_steps <= 0:
        raise ValueError("epochs, steps-per-epoch and val-steps must be positive")
    if args.selection_metric == "full_mae" and args.full_val_every <= 0:
        raise ValueError("selection-metric=full_mae requires full-val-every > 0")
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.radiological_depth and not args.physics_priors:
        args.physics_priors = True
        print("--radiological-depth enables --physics-priors")
    seed_everything(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Output: {args.output_dir}")

    train_dataset, train_loader, validation_dataset, validation_loader = make_loaders(args)
    print(f"Train control points: {len(train_dataset)}")
    print(f"Validation control points: {len(validation_dataset)}")

    model_config = ModelConfig(
        in_channels=(
            13
            if args.radiological_depth
            else 11 if args.physics_priors else (7 if args.include_density else 6)
        ),
        base_channels=args.base_channels,
        levels=args.levels,
        blocks_per_level=args.blocks_per_level,
        dropout=args.dropout,
        physics_priors=args.physics_priors,
        radiological_depth=args.radiological_depth,
        scale_head=args.scale_head,
        pb_head=args.pb_dose_dir is not None and args.pb_distill_weight > 0,
        deep_supervision=args.deep_supervision,
    )
    model: torch.nn.Module = PhotonDoseUNet3D(model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameter_count:,}")

    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        state = initial["model"].copy()
        current_stem = model.state_dict()["stem.weight"]
        old_stem = state["stem.weight"]
        if old_stem.shape != current_stem.shape:
            if (
                old_stem.shape[0] == current_stem.shape[0]
                and old_stem.shape[2:] == current_stem.shape[2:]
                and old_stem.shape[1] < current_stem.shape[1]
            ):
                expanded = torch.zeros_like(current_stem)
                expanded[:, : old_stem.shape[1]] = old_stem
                state["stem.weight"] = expanded
                print(
                    f"Expanded input stem {old_stem.shape[1]} -> "
                    f"{current_stem.shape[1]} channels with zero-initialized priors"
                )
            else:
                raise ValueError(
                    f"cannot adapt stem {tuple(old_stem.shape)} -> {tuple(current_stem.shape)}"
                )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"init checkpoint has unexpected keys: {unexpected[:5]}")
        if missing:
            allowed_new = all(
                key.startswith(("scale_head.", "pb_head.")) for key in missing
            )
            if not allowed_new:
                raise ValueError(f"init checkpoint is missing keys: {missing[:5]}")
            print(f"Initialized {len(missing)} new head parameters from scratch")
        print(f"Initialized model weights from {args.init_checkpoint}")

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp and device.type == "cuda")
    criterion = DoseLoss(
        LossConfig(
            high_dose_weight=args.high_dose_weight,
            gradient_weight=args.gradient_loss_weight,
            official_mae_weight=args.official_mae_weight,
            idd_surrogate_weight=args.idd_surrogate_weight,
            scale_weight=args.scale_loss_weight,
            out_of_field_weight=args.out_of_field_weight,
            out_of_field_threshold=args.out_of_field_threshold,
            pb_distill_weight=args.pb_distill_weight,
            deep_supervision_weight=args.deep_supervision_weight if args.deep_supervision else 0.0,
        )
    )

    start_epoch = 1
    best_metric = float("inf")
    best_patch_metric = float("inf")
    best_full_metric = float("inf")
    best_idd_metric = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))
        best_patch_metric = float(checkpoint.get("best_patch_metric", best_metric))
        best_full_metric = float(checkpoint.get("best_full_metric", best_metric))
        best_idd_metric = float(checkpoint.get("best_idd_metric", float("inf")))
        print(f"Resumed from epoch {start_epoch - 1}: {args.resume}")

    if args.compile:
        model = torch.compile(model)

    config_path = args.output_dir / "config.json"
    config_payload = vars(args).copy()
    config_payload.update(
        {
            "manifest": str(args.manifest),
            "splits": str(args.splits),
            "output_dir": str(args.output_dir),
            "resume": str(args.resume) if args.resume else None,
            "init_checkpoint": (
                str(args.init_checkpoint) if args.init_checkpoint else None
            ),
            "model": model_config.to_dict(),
            "parameter_count": parameter_count,
        }
    )
    config_path.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    metrics_path = args.output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        epoch_started = time.perf_counter()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args
        )
        validation_metrics = validate_patches(
            model, validation_loader, criterion, device, args
        )
        full_metrics = None
        if args.full_val_every > 0 and epoch % args.full_val_every == 0:
            full_metrics = validate_full_volumes(
                model, validation_dataset, device, args
            )
        scheduler.step()

        patch_metric = validation_metrics["masked_mae"]
        patch_improved = math.isfinite(patch_metric) and patch_metric < best_patch_metric
        if patch_improved:
            best_patch_metric = patch_metric

        full_metric = (
            full_metrics["masked_mae"] if full_metrics is not None else float("nan")
        )
        full_improved = math.isfinite(full_metric) and full_metric < best_full_metric
        if full_improved:
            best_full_metric = full_metric

        idd_metric = (
            full_metrics["idd_distance"] if full_metrics is not None else float("nan")
        )
        idd_improved = math.isfinite(idd_metric) and idd_metric < best_idd_metric
        if idd_improved:
            best_idd_metric = idd_metric

        candidate_metric = (
            patch_metric if args.selection_metric == "patch_mae" else full_metric
        )
        improved = math.isfinite(candidate_metric) and candidate_metric < best_metric
        if improved:
            best_metric = candidate_metric

        payload = {
            "epoch": epoch,
            "seconds": time.perf_counter() - epoch_started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
            "full_validation": full_metrics,
            "selection_metric": args.selection_metric,
            "best_metric": best_metric,
            "best_patch_metric": best_patch_metric,
            "best_full_metric": best_full_metric,
            "best_idd_metric": best_idd_metric,
        }
        append_metrics(metrics_path, payload)
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_metric,
            best_patch_metric,
            best_full_metric,
            best_idd_metric,
            model_config,
            args,
        )
        if patch_improved:
            save_checkpoint(
                args.output_dir / "best_patch.pt",
                model, optimizer, scheduler, scaler, epoch,
                best_metric, best_patch_metric, best_full_metric,
                best_idd_metric,
                model_config, args,
            )
        if full_improved:
            save_checkpoint(
                args.output_dir / "best_full.pt",
                model, optimizer, scheduler, scaler, epoch,
                best_metric, best_patch_metric, best_full_metric,
                best_idd_metric,
                model_config, args,
            )
        if idd_improved:
            save_checkpoint(
                args.output_dir / "best_idd.pt",
                model, optimizer, scheduler, scaler, epoch,
                best_metric, best_patch_metric, best_full_metric,
                best_idd_metric,
                model_config, args,
            )
        if improved:
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metric,
                best_patch_metric,
                best_full_metric,
                best_idd_metric,
                model_config,
                args,
            )
        print(
            f"epoch={epoch} val_loss={validation_metrics['loss']:.6f} "
            f"val_masked_mae={validation_metrics['masked_mae']:.5f} "
            f"best={best_metric:.5f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
