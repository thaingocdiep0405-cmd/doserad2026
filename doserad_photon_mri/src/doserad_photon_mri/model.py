from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))
        return x + residual


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 6
    base_channels: int = 12
    levels: int = 4
    blocks_per_level: int = 2
    dropout: float = 0.0
    physics_priors: bool = False
    auxiliary_density: bool = False
    deep_supervision: bool = False

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class PhotonDoseUNet3D(nn.Module):
    """Compact residual 3D U-Net for per-control-point dose prediction."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        if config.levels < 2:
            raise ValueError("levels must be at least 2")
        self.config = config
        features = [config.base_channels * (2**level) for level in range(config.levels)]

        self.stem = nn.Conv3d(config.in_channels, features[0], kernel_size=3, padding=1)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for level, channels in enumerate(features):
            blocks = [ResidualBlock3D(channels, channels, config.dropout)]
            blocks.extend(
                ResidualBlock3D(channels, channels, config.dropout)
                for _ in range(config.blocks_per_level - 1)
            )
            self.encoders.append(nn.Sequential(*blocks))
            if level < len(features) - 1:
                self.downsamples.append(
                    nn.Conv3d(channels, features[level + 1], kernel_size=3, stride=2, padding=1)
                )

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(len(features) - 2, -1, -1):
            self.upsamples.append(
                nn.ConvTranspose3d(features[level + 1], features[level], kernel_size=2, stride=2)
            )
            decoder_blocks: list[nn.Module] = [
                ResidualBlock3D(features[level] * 2, features[level], config.dropout)
            ]
            decoder_blocks.extend(
                ResidualBlock3D(features[level], features[level], config.dropout)
                for _ in range(config.blocks_per_level - 1)
            )
            self.decoders.append(nn.Sequential(*decoder_blocks))

        self.output_norm = nn.GroupNorm(_group_count(features[0]), features[0])
        self.output = nn.Conv3d(features[0], 1, kernel_size=1)
        nn.init.constant_(self.output.bias, -5.0)
        self.density_output = (
            nn.Conv3d(features[0], 1, kernel_size=1)
            if config.auxiliary_density
            else None
        )
        if self.density_output is not None:
            nn.init.constant_(self.density_output.bias, 0.5)

        if config.deep_supervision:
            self.ds_heads = nn.ModuleList()
            n_decoder_levels = len(features) - 1
            for ds_idx in range(1, min(n_decoder_levels, 3)):
                ds_channels = features[ds_idx]
                head = nn.Sequential(
                    nn.GroupNorm(_group_count(ds_channels), ds_channels),
                    nn.SiLU(inplace=True),
                    nn.Conv3d(ds_channels, 1, kernel_size=1),
                )
                nn.init.constant_(head[-1].bias, -5.0)
                self.ds_heads.append(head)
        else:
            self.ds_heads = nn.ModuleList()

    def _run_backbone(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Size]:
        target_shape = x.shape[2:]
        x = self.stem(x)
        skips: list[torch.Tensor] = []
        for level, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if level < len(self.downsamples):
                x = self.downsamples[level](x)

        decoder_outputs: list[torch.Tensor] = []
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips[:-1])):
            x = upsample(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
            decoder_outputs.append(x)

        final_features = F.silu(self.output_norm(x))
        return final_features, decoder_outputs, target_shape

    def _build_dose_output(
        self,
        final_features: torch.Tensor,
        decoder_outputs: list[torch.Tensor],
        target_shape: torch.Size,
    ) -> torch.Tensor | list[torch.Tensor]:
        main_out = F.softplus(self.output(final_features))

        if self.config.deep_supervision and self.training and self.ds_heads:
            outputs = [main_out]
            for ds_idx, head in enumerate(self.ds_heads):
                ds_feat = decoder_outputs[-(ds_idx + 2)]
                ds_out = F.softplus(head(ds_feat))
                ds_out = F.interpolate(
                    ds_out, size=target_shape, mode="trilinear", align_corners=False
                )
                outputs.append(ds_out)
            return outputs

        return main_out

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        final_features, decoder_outputs, target_shape = self._run_backbone(x)
        return self._build_dose_output(final_features, decoder_outputs, target_shape)

    def forward_with_auxiliary(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | None]:
        final_features, decoder_outputs, target_shape = self._run_backbone(x)
        dose = self._build_dose_output(final_features, decoder_outputs, target_shape)
        density = (
            F.softplus(self.density_output(final_features))
            if self.density_output is not None
            else None
        )
        return dose, density
