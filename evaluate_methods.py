"""Unified Stage A evaluation entry point for fixed HGCR-PPO instances."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

from instance_manager import SIZES, SPLITS, ensure_fixed_dataset, load_fixed_instances
from src.baselines.heuristics import POLICIES, run_heuristic


RESULT_DIR = Path("data/results/stage_A")
BASELINE_CSV = RESULT_DIR / "stage_A_baselines.csv"
SUMMARY_CSV = RESULT_DIR / "stage_A_summary.csv"
METRICS = [
    "Cmax_roll",
    "average_completion_time",
    "average_waiting_time",
    "machine_utilization",
    "load_balance_std",
    "split_task_ratio",
    "total_split_count",
    "inference_time",
]
ROW_FIELDS = [
    "method",
    "size",
    "split",
    "seed",
    "instance_id",
    "Cmax_roll",
    "average_completion_time",
    "average_waiting_time",
    "machine_utilization",
    "load_balance_std",
    "split_task_ratio",
    "total_split_count",
    "inference_time",
    "candidate_mode",
    "split_rule",
    "notes",
]


def evaluate_size_split(size: str, split: str, seed: int = 42) -> List[Dict]:
    ensure_fixed_dataset([size], [split])
    rows: List[Dict] = []
    for instance in load_fixed_instances(size, split):
        for method in POLICIES:
            start = time.perf_counter()
            result = run_heuristic(instance, method, seed=seed)
            inference_time = time.perf_counter() - start
            rows.append(
                {
                    "method": method,
                    "size": size,
                    "split": split,
                    "seed": getattr(instance, "seed", ""),
                    "instance_id": getattr(instance, "instance_id", instance.name),
                    **result.metrics,
                    "inference_time": inference_time,
                    "candidate_mode": "all_eligible",
                    "split_rule": "env_inverse_processing_time",
                    "notes": "stage_A_fixed_instances",
                }
            )
    return rows


def write_baselines(rows: Iterable[Dict], output_path: Path = BASELINE_CSV) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: Iterable[Dict], output_path: Path = SUMMARY_CSV) -> List[Dict]:
    rows = list(rows)
    grouped: Dict[tuple[str, str, str], List[Dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["size"], row["split"]), []).append(row)

    summary_rows = []
    for (method, size, split), group in sorted(grouped.items()):
        out = {"method": method, "size": size, "split": split, "num_instances": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summary_rows.append(out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "size", "split", "num_instances"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def print_summary(summary_rows: Iterable[Dict]) -> None:
    print("method,size,split,Cmax_roll_mean,Cmax_roll_std,machine_utilization_mean")
    for row in summary_rows:
        print(
            f"{row['method']},{row['size']},{row['split']},"
            f"{row['Cmax_roll_mean']:.3f},{row['Cmax_roll_std']:.3f},"
            f"{row['machine_utilization_mean']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = evaluate_size_split(args.size, args.split, seed=args.seed)
    write_baselines(rows)
    summary_rows = write_summary(rows)
    print_summary(summary_rows)
    print(f"Saved rows to {BASELINE_CSV}")
    print(f"Saved summary to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
