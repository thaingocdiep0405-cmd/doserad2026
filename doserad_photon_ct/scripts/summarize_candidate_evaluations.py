#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    ("masked_beam_mae", "MAE"),
    ("idd_curve_distance", "IDD"),
    ("normalized_rmse", "NRMSE"),
)


def paired_summary(
    candidate: dict,
    reference: dict,
    key: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    reference_by_patient = {
        item["patient_id"]: float(item[key]) for item in reference["patient_metrics"]
    }
    differences = np.asarray(
        [
            float(item[key]) - reference_by_patient[item["patient_id"]]
            for item in candidate["patient_metrics"]
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, differences.size, size=(bootstrap_samples, differences.size)
    )
    bootstrap_means = differences[indices].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "relative_change_percent": float(
            100.0 * differences.mean() / float(reference[f"mean_{key}"])
        ),
        "paired_ci_95": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "patient_win_rate": float(np.mean(differences < 0.0)),
        "patients": int(differences.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path("artifacts/task1_candidate_evaluations")
    )
    parser.add_argument(
        "--reference", default="v2_reference", help="Reference JSON stem"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/task1_candidate_summary.json")
    )
    args = parser.parse_args()

    evaluations = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_dir.glob("*.json"))
    }
    if args.reference not in evaluations:
        raise FileNotFoundError(f"missing reference {args.reference}.json")
    reference = evaluations[args.reference]
    result = {"reference": args.reference, "candidates": {}}
    for candidate_index, (name, evaluation) in enumerate(evaluations.items()):
        metrics = {
            key: float(evaluation[f"mean_{key}"]) for key, _ in METRICS
        }
        relative_mae = metrics["masked_beam_mae"] / float(
            reference["mean_masked_beam_mae"]
        )
        relative_idd = metrics["idd_curve_distance"] / float(
            reference["mean_idd_curve_distance"]
        )
        result["candidates"][name] = {
            "checkpoint": evaluation["checkpoint"],
            "metrics": metrics,
            "mae_idd_geometric_relative_score": float(
                np.sqrt(relative_mae * relative_idd)
            ),
            "paired_vs_reference": {
                key: paired_summary(
                    evaluation,
                    reference,
                    key,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=2026 + 10 * candidate_index + metric_index,
                )
                for metric_index, (key, _) in enumerate(METRICS)
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("candidate\tMAE\tIDD\tNRMSE\trelative_score")
    for name, item in sorted(
        result["candidates"].items(),
        key=lambda pair: pair[1]["mae_idd_geometric_relative_score"],
    ):
        metrics = item["metrics"]
        print(
            f"{name}\t{metrics['masked_beam_mae']:.6f}\t"
            f"{metrics['idd_curve_distance']:.6f}\t"
            f"{metrics['normalized_rmse']:.6f}\t"
            f"{item['mae_idd_geometric_relative_score']:.4f}"
        )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
