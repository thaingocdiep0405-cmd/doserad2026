from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LossConfig:
    high_dose_weight: float = 4.0
    high_dose_threshold: float = 0.1
    smooth_l1_beta: float = 0.02
    gradient_weight: float = 0.1
    official_mae_weight: float = 0.0
    idd_surrogate_weight: float = 0.0
    scale_weight: float = 0.0
    out_of_field_weight: float = 0.0
    out_of_field_threshold: float = 0.02
    pb_distill_weight: float = 0.0
    deep_supervision_weight: float = 0.0


def _spatial_gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for dimension in (2, 3, 4):
        pred_diff = torch.diff(prediction, dim=dimension)
        target_diff = torch.diff(target, dim=dimension)
        losses.append(F.l1_loss(pred_diff, target_diff))
    return torch.stack(losses).mean()


def _official_masked_mae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    maximum: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    values = []
    for pred_item, target_item, item_maximum in zip(prediction, target, maximum):
        denominator = item_maximum.reshape(()).clamp_min(1.0e-8)
        mask = target_item >= threshold * denominator
        if mask.any():
            values.append(torch.abs(pred_item[mask] - target_item[mask]).mean() / denominator)
    return torch.stack(values).mean() if values else prediction.new_zeros(())


def _directional_idd_surrogate(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gantry_angles_deg: torch.Tensor | None,
    target_maximum: torch.Tensor,
) -> torch.Tensor:
    """Differentiable patch-level projection loss aligned with the official IDD."""
    if gantry_angles_deg is None:
        return prediction.new_zeros(())
    prediction_plane = prediction.float().sum(dim=2)
    target_plane = target.float().sum(dim=2)
    angles = torch.deg2rad(
        gantry_angles_deg.to(device=prediction.device, dtype=torch.float32).reshape(-1)
    )
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    affine = torch.zeros(
        (prediction.shape[0], 2, 3), device=prediction.device, dtype=torch.float32
    )
    affine[:, 0, 0] = cosine
    affine[:, 0, 1] = -sine
    affine[:, 1, 0] = sine
    affine[:, 1, 1] = cosine
    grid = F.affine_grid(affine, prediction_plane.shape, align_corners=False)
    pred_aligned = F.grid_sample(
        prediction_plane, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    target_aligned = F.grid_sample(
        target_plane, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    pred_curve = pred_aligned.sum(dim=2)
    target_curve = target_aligned.sum(dim=2)
    denominator = target_curve.amax(dim=(1, 2), keepdim=True)
    patch_peak = target.amax(dim=(1, 2, 3, 4))
    global_peak = target_maximum.reshape(-1).to(patch_peak).clamp_min(1.0e-8)
    valid = (patch_peak >= 0.01 * global_peak) & (denominator.reshape(-1) > 1.0e-6)
    if not valid.any():
        return prediction.new_zeros(())
    denominator = denominator[valid].clamp_min(1.0e-6)
    pred_normalized = (pred_curve[valid] / denominator).clamp(max=5.0)
    target_normalized = target_curve[valid] / denominator
    return F.smooth_l1_loss(pred_normalized, target_normalized, beta=0.02)


def _out_of_field_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    maximum: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Mean predicted dose in voxels the reference leaves essentially cold.

    The official masked MAE ignores voxels below 10% of the beam maximum,
    but the IDD curve integrates the whole volume, so dose hallucinated in
    near-zero regions costs leaderboard rank without moving the MAE. This
    term pushes those voxels toward zero, normalized by the beam maximum.
    """
    values = []
    for pred_item, target_item, item_maximum in zip(prediction, target, maximum):
        denominator = item_maximum.reshape(()).clamp_min(1.0e-8)
        mask = target_item < threshold * denominator
        if mask.any():
            values.append(pred_item[mask].clamp_min(0.0).mean() / denominator)
    return torch.stack(values).mean() if values else prediction.new_zeros(())


def _dose_scale_loss(
    prediction: torch.Tensor, target: torch.Tensor, maximum: torch.Tensor
) -> torch.Tensor:
    values = []
    for pred_item, target_item, item_maximum in zip(prediction, target, maximum):
        threshold = 0.1 * item_maximum.reshape(()).clamp_min(1.0e-8)
        mask = target_item >= threshold
        if mask.any():
            target_mean = target_item[mask].mean().clamp_min(1.0e-8)
            values.append(torch.abs(pred_item[mask].mean() / target_mean - 1.0))
    return torch.stack(values).mean() if values else prediction.new_zeros(())


class DoseLoss(nn.Module):
    def __init__(self, config: LossConfig = LossConfig()):
        super().__init__()
        self.config = config

    def _single_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        maximum: torch.Tensor,
        body_mask: torch.Tensor | None,
        gantry_angles_deg: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if body_mask is not None and body_mask.any():
            full = F.smooth_l1_loss(
                prediction[body_mask],
                target[body_mask],
                beta=self.config.smooth_l1_beta,
                reduction="mean",
            )
        else:
            full = F.smooth_l1_loss(
                prediction, target, beta=self.config.smooth_l1_beta, reduction="mean"
            )
        high_mask = target >= self.config.high_dose_threshold * maximum
        if high_mask.any():
            high = torch.abs(prediction - target)[high_mask].mean()
        else:
            high = prediction.new_zeros(())
        gradient = _spatial_gradient_loss(prediction, target)
        official_mae = _official_masked_mae_loss(
            prediction,
            target,
            maximum,
            self.config.high_dose_threshold,
        )
        idd_surrogate = _directional_idd_surrogate(
            prediction, target, gantry_angles_deg, maximum
        )
        scale = _dose_scale_loss(prediction, target, maximum)
        out_of_field = _out_of_field_loss(
            prediction, target, maximum, self.config.out_of_field_threshold
        )
        total = (
            full
            + self.config.high_dose_weight * high
            + self.config.gradient_weight * gradient
            + self.config.official_mae_weight * official_mae
            + self.config.idd_surrogate_weight * idd_surrogate
            + self.config.scale_weight * scale
            + self.config.out_of_field_weight * out_of_field
        )
        return total, {
            "full": full,
            "high": high,
            "gradient": gradient,
            "official_mae": official_mae,
            "idd_surrogate": idd_surrogate,
            "scale": scale,
            "out_of_field": out_of_field,
        }

    def _pb_distillation_loss(
        self,
        pb_prediction: torch.Tensor | None,
        pb_target: torch.Tensor | None,
        pb_valid: torch.Tensor | None,
        body_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Auxiliary-head loss toward the offline pencil-beam dose.

        Only samples that actually have a precomputed PB volume contribute
        (``pb_valid`` flags them); the PB engine is imperfect, so this signal
        feeds a separate head rather than the main output.
        """
        if pb_prediction is None or pb_target is None:
            return None
        if pb_valid is not None:
            keep = pb_valid.to(device=pb_prediction.device, dtype=torch.bool).reshape(-1)
            if not keep.any():
                return pb_prediction.new_zeros(())
            pb_prediction = pb_prediction[keep]
            pb_target = pb_target[keep]
            body_mask = body_mask[keep] if body_mask is not None else None
        if body_mask is not None and body_mask.any():
            return F.smooth_l1_loss(
                pb_prediction[body_mask],
                pb_target[body_mask],
                beta=self.config.smooth_l1_beta,
            )
        return F.smooth_l1_loss(
            pb_prediction, pb_target, beta=self.config.smooth_l1_beta
        )

    def forward(
        self,
        prediction: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor,
        target_max: torch.Tensor | None = None,
        body_mask: torch.Tensor | None = None,
        gantry_angles_deg: torch.Tensor | None = None,
        pb_prediction: torch.Tensor | None = None,
        pb_target: torch.Tensor | None = None,
        pb_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if target_max is None:
            maximum = target.amax(dim=(2, 3, 4), keepdim=True)
        else:
            maximum = target_max.to(device=target.device, dtype=target.dtype).reshape(
                -1, 1, 1, 1, 1
            )

        if isinstance(prediction, list):
            main_pred = prediction[0]
            total, components = self._single_loss(
                main_pred, target, maximum, body_mask, gantry_angles_deg
            )
            if self.config.deep_supervision_weight > 0 and len(prediction) > 1:
                ds_losses = []
                for ds_pred in prediction[1:]:
                    ds_loss, _ = self._single_loss(
                        ds_pred, target, maximum, body_mask, gantry_angles_deg
                    )
                    ds_losses.append(ds_loss)
                ds_mean = torch.stack(ds_losses).mean()
                total = total + self.config.deep_supervision_weight * ds_mean
                components["deep_supervision"] = ds_mean
        else:
            total, components = self._single_loss(
                prediction, target, maximum, body_mask, gantry_angles_deg
            )

        if self.config.pb_distill_weight > 0:
            pb_loss = self._pb_distillation_loss(
                pb_prediction, pb_target, pb_valid, body_mask
            )
            if pb_loss is not None:
                total = total + self.config.pb_distill_weight * pb_loss
                components["pb_distill"] = pb_loss
        return total, components


@torch.no_grad()
def masked_beam_mae_tensor(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_max: torch.Tensor | None = None,
) -> torch.Tensor:
    """Official masked beam MAE definition, averaged over valid batch items."""
    values = []
    maxima = (
        target.amax(dim=(1, 2, 3, 4))
        if target_max is None
        else target_max.to(device=target.device, dtype=target.dtype).reshape(-1)
    )
    for pred_item, target_item, maximum in zip(prediction, target, maxima):
        if float(maximum) <= 0:
            continue
        mask = target_item >= 0.1 * maximum
        if mask.any():
            values.append(torch.abs(pred_item[mask] - target_item[mask]).mean() / maximum)
    if not values:
        return prediction.new_tensor(float("nan"))
    return torch.stack(values).mean()


@torch.no_grad()
def normalized_rmse_tensor(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_max: torch.Tensor | None = None,
) -> torch.Tensor:
    denominator = (
        target.amax(dim=(1, 2, 3, 4))
        if target_max is None
        else target_max.to(device=target.device, dtype=target.dtype).reshape(-1)
    ).clamp_min(1.0e-8)
    rmse = torch.sqrt(torch.mean((prediction - target) ** 2, dim=(1, 2, 3, 4)))
    return (rmse / denominator).mean()
