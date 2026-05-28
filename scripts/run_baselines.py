"""Run heuristic baselines for small, medium, and large instances."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.heuristics import POLICIES, run_heuristic
from src.instances.instance_generator import generate_instance


METRIC_COLUMNS = [
    "Cmax_roll",
    "average_completion_time",
    "average_waiting_time",
    "machine_utilization",
    "load_balance_std",
    "split_task_ratio",
    "total_split_count",
]


def _format_metrics_row(size: str, heuristic: str, metrics: dict) -> str:
    values = [
        size,
        heuristic,
        f"{metrics['Cmax_roll']:.2f}",
        f"{metrics['average_completion_time']:.2f}",
        f"{metrics['average_waiting_time']:.2f}",
        f"{metrics['machine_utilization']:.3f}",
        f"{metrics['load_balance_std']:.2f}",
        f"{metrics['split_task_ratio']:.3f}",
        f"{metrics['total_split_count']:.0f}",
    ]
    widths = [8, 18, 12, 24, 22, 21, 18, 18, 17]
    return " ".join(value.ljust(width) for value, width in zip(values, widths))


def run(seed: int = 42, results_dir: str = "data/results") -> None:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    header = _format_metrics_row(
        "size",
        "heuristic",
        {
            "Cmax_roll": 0.0,
            "average_completion_time": 0.0,
            "average_waiting_time": 0.0,
            "machine_utilization": 0.0,
            "load_balance_std": 0.0,
            "split_task_ratio": 0.0,
            "total_split_count": 0.0,
        },
    )
    header = (
        "size     heuristic          Cmax_roll    average_completion_time "
        "average_waiting_time   machine_utilization load_balance_std  "
        "split_task_ratio  total_split_count"
    )
    print(header)
    print("-" * len(header))

    saved_gantt = None
    for size in ["small", "medium", "large"]:
        instance = generate_instance(size=size, seed=seed)
        for heuristic_name in POLICIES:
            result = run_heuristic(instance, heuristic_name, seed=seed)
            print(_format_metrics_row(size, heuristic_name, result.metrics))

            if size == "small":
                gantt_path = Path(results_dir) / f"gantt_{size}_{heuristic_name}.png"
                result.env.render_gantt(str(gantt_path))
                saved_gantt = saved_gantt or gantt_path

    if saved_gantt:
        print(f"\nSaved Gantt charts under: {Path(results_dir).resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="data/results")
    args = parser.parse_args()
    run(seed=args.seed, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
