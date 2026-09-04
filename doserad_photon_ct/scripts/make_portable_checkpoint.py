#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def portable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create inference-only portable checkpoint")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    output = {
        "format_version": int(checkpoint.get("format_version", 1)),
        "epoch": int(checkpoint.get("epoch", 0)),
        "model_config": portable(checkpoint["model_config"]),
        "training_config": portable(checkpoint["training_config"]),
        "model": checkpoint["model"],
    }
    if "blend" in checkpoint:
        output["blend"] = portable(checkpoint["blend"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"Wrote portable checkpoint {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
