"""One-command Stage A runner for fixed-data heuristic evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from evaluate_methods import evaluate_size_split, write_baselines, write_summary
from instance_manager import SIZES, ensure_fixed_dataset
from src.baselines.heuristics import run_heuristic
from src.visualization import plot_gantt
from instance_manager import load_fixed_instances


RESULT_DIR = Path("data/results/stage_A")


def _save_check_gantts(size: str) -> None:
    instances = load_fixed_instances(size, "test")
    if not instances:
        return
    instance = instances[0]
    for method in ["FIFO", "GreedyECT"]:
        result = run_heuristic(instance, method)
        path = RESULT_DIR / "gantt" / f"stage_A_{size}_{method}.png"
        plot_gantt(result.env, str(path))


def run_stage_a(sizes: List[str]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_fixed_dataset(sizes, ["train", "val", "test"])
    all_rows = []
    for size in sizes:
        all_rows.extend(evaluate_size_split(size, "test"))
        _save_check_gantts(size)
    write_baselines(all_rows)
    write_summary(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=["small", "medium", "large"])
    args = parser.parse_args()
    run_stage_a(args.sizes)
    print(f"Stage A results saved under {RESULT_DIR}")


if __name__ == "__main__":
    main()
