#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def portable(value):
    """Remove Python-version-specific objects from checkpoint metadata."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linearly blend two compatible model checkpoints."
    )
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument(
        "--first-weight",
        type=float,
        required=True,
        help="Weight for the first checkpoint; second receives 1-weight",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.first_weight <= 1.0:
        raise ValueError("first-weight must be in [0, 1]")
    first = torch.load(args.first, map_location="cpu", weights_only=False)
    second = torch.load(args.second, map_location="cpu", weights_only=False)
    if first["model_config"] != second["model_config"]:
        raise ValueError("model configurations do not match")
    if first["model"].keys() != second["model"].keys():
        raise ValueError("model state dictionaries do not match")

    alpha = args.first_weight
    blended = first.copy()
    blended["model"] = {
        key: first["model"][key].float().mul(alpha).add(
            second["model"][key].float(), alpha=1.0 - alpha
        ).to(first["model"][key].dtype)
        for key in first["model"]
    }
    # Optimizer/scheduler states do not correspond to blended parameters and
    # must never be used for resume training.
    blended.pop("optimizer", None)
    blended.pop("scheduler", None)
    blended.pop("scaler", None)
    blended["blend"] = {
        "first": str(args.first.resolve()),
        "second": str(args.second.resolve()),
        "first_weight": alpha,
        "second_weight": 1.0 - alpha,
    }
    for key in list(blended):
        if key != "model":
            blended[key] = portable(blended[key])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blended, args.output)
    print(f"Wrote {args.output} (first={alpha:.2f}, second={1.0-alpha:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
