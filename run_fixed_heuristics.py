"""Run heuristic baselines on fixed test instances.

This script is intentionally separate from ``run_baselines.py``. It never
generates random one-off instances; all comparisons are loaded from
``data/instances/`` so PPO, GNN-PPO, and heuristic baselines can share the same
test sets.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

from dataset_manager import load_dataset
from src.baselines.heuristics import POLICIES, run_heuristic


SIZES = ["small", "medium", "large"]
METRIC_KEYS = [
    "Cmax_roll",
    "average_completion_time",
    "average_waiting_time",
    "machine_utilization",
    "load_balance_std",
    "split_task_ratio",
    "total_split_count",
]
OUTPUT_PATH = Path("data/results/ppo/heuristic_baselines_fixed_instances.csv")
GANTT_DIR = Path("data/results/ppo/gantt/heuristics_fixed")


def _aggregate(metric_rows: List[Dict[str, float]]) -> Dict[str, float]:
    aggregated: Dict[str, float] = {}
    for key in METRIC_KEYS:
        values = [float(row[key]) for row in metric_rows]
        aggregated[f"{key}_mean"] = mean(values)
        aggregated[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return aggregated


def _load_fixed_test_instances(size: str) -> List:
    try:
        return load_dataset(size=size, split="test")
    except FileNotFoundError as exc:
        raise SystemExit(
            "Fixed test instances are missing. Please run: "
            "python dataset_manager.py --generate_all"
        ) from exc


def run_fixed_heuristics(seed: int = 42) -> List[Dict[str, float]]:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GANTT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float]] = []

    for size in SIZES:
        instances = _load_fixed_test_instances(size)
        representative_instance = instances[0]

        for heuristic in POLICIES:
            metric_rows = []
            inference_times = []
            representative_env = None

            for idx, instance in enumerate(instances):
                start = time.perf_counter()
                result = run_heuristic(instance, heuristic, seed=seed)
                inference_times.append(time.perf_counter() - start)
                metric_rows.append(result.metrics)
                if idx == 0:
                    representative_env = result.env

            aggregated = _aggregate(metric_rows)
            row = {
                "size": size,
                "heuristic": heuristic,
                "num_instances": len(instances),
                **aggregated,
                "inference_time_per_instance_mean": mean(inference_times),
            }
            rows.append(row)

            if representative_env is not None:
                gantt_path = GANTT_DIR / f"gantt_fixed_{size}_{heuristic}.png"
                representative_env.render_gantt(str(gantt_path))

    _write_rows(rows)
    return rows


def _write_rows(rows: Iterable[Dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = [
        "size",
        "heuristic",
        "num_instances",
        "Cmax_roll_mean",
        "Cmax_roll_std",
        "average_completion_time_mean",
        "average_completion_time_std",
        "average_waiting_time_mean",
        "average_waiting_time_std",
        "machine_utilization_mean",
        "machine_utilization_std",
        "load_balance_std_mean",
        "load_balance_std_std",
        "split_task_ratio_mean",
        "split_task_ratio_std",
        "total_split_count_mean",
        "total_split_count_std",
        "inference_time_per_instance_mean",
    ]
    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: Iterable[Dict[str, float]]) -> None:
    header = (
        "size heuristic Cmax_roll_mean average_completion_time_mean "
        "average_waiting_time_mean machine_utilization_mean "
        "split_task_ratio_mean total_split_count_mean"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['size']:<6} {row['heuristic']:<18} "
            f"{row['Cmax_roll_mean']:<15.2f} "
            f"{row['average_completion_time_mean']:<29.2f} "
            f"{row['average_waiting_time_mean']:<26.2f} "
            f"{row['machine_utilization_mean']:<24.3f} "
            f"{row['split_task_ratio_mean']:<21.3f} "
            f"{row['total_split_count_mean']:.2f}"
        )


def main() -> None:
    rows = run_fixed_heuristics()
    _print_summary(rows)
    print(f"\nSaved fixed-instance heuristic metrics to: {OUTPUT_PATH}")
    print(f"Saved representative Gantt charts to: {GANTT_DIR}")


if __name__ == "__main__":
    main()
