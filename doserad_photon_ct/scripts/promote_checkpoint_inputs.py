#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from blend_checkpoints import portable


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote a checkpoint to a compatible wider input stem"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="Checkpoint defining the target model/training configuration",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = torch.load(args.input, map_location="cpu", weights_only=False)
    template = torch.load(args.template, map_location="cpu", weights_only=False)
    promoted_state = {}
    for key, target_tensor in template["model"].items():
        if key not in source["model"]:
            raise ValueError(f"source checkpoint is missing {key}")
        source_tensor = source["model"][key]
        if source_tensor.shape == target_tensor.shape:
            promoted_state[key] = source_tensor
            continue
        if (
            key == "stem.weight"
            and source_tensor.shape[0] == target_tensor.shape[0]
            and source_tensor.shape[2:] == target_tensor.shape[2:]
            and source_tensor.shape[1] < target_tensor.shape[1]
        ):
            expanded = torch.zeros_like(target_tensor)
            expanded[:, : source_tensor.shape[1]] = source_tensor
            promoted_state[key] = expanded
            continue
        raise ValueError(
            f"cannot promote {key}: {tuple(source_tensor.shape)} -> "
            f"{tuple(target_tensor.shape)}"
        )

    promoted = {
        "format_version": int(source.get("format_version", 1)),
        "epoch": int(source.get("epoch", 0)),
        "model_config": portable(template["model_config"]),
        "training_config": portable(template["training_config"]),
        "model": promoted_state,
        "promotion": {
            "source": str(args.input.resolve()),
            "template": str(args.template.resolve()),
            "source_in_channels": int(source["model"]["stem.weight"].shape[1]),
            "target_in_channels": int(template["model"]["stem.weight"].shape[1]),
            "new_channels_initialized_to_zero": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(promoted, args.output)
    print(
        f"Promoted {args.input} from "
        f"{promoted['promotion']['source_in_channels']} to "
        f"{promoted['promotion']['target_in_channels']} channels: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
