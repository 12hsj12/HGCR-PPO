"""One-command Stage A runner for fixed-data heuristic evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from check_split_effect import evaluate_split_effect, write_split_summary
from evaluate_methods import evaluate_size_split, write_baselines, write_summary
from instance_manager import SIZES, ensure_fixed_dataset
from src.baselines.heuristics import run_heuristic
from src.visualization import plot_gantt
from instance_manager import load_fixed_instances


RESULT_DIR = Path("data/results/stage_A")
REPORT_PATH = RESULT_DIR / "stage_A_report.md"


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
    split_rows = []
    for size in sizes:
        all_rows.extend(evaluate_size_split(size, "test"))
        split_rows.extend(evaluate_split_effect(size, "test"))
        _save_check_gantts(size)
    write_baselines(all_rows)
    summary_rows = write_summary(all_rows)
    split_summary_rows = write_split_summary(split_rows, merge_existing=False)
    write_stage_a_report(summary_rows, split_summary_rows)


def _fmt(value: str | float, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_stage_a_report(summary_rows: List[dict], split_summary_rows: List[dict]) -> None:
    best_by_size = {}
    for row in summary_rows:
        size = row["size"]
        current = best_by_size.get(size)
        if current is None or float(row["Cmax_roll_mean"]) < float(current["Cmax_roll_mean"]):
            best_by_size[size] = row

    invalid_baseline = [row for row in summary_rows if float(row.get("valid_ratio", 0.0)) < 1.0]
    invalid_split = [row for row in split_summary_rows if float(row.get("valid_ratio", 0.0)) < 1.0]

    lines = [
        "# Stage A Report",
        "",
        "## Baseline Summary",
        "",
        "| size | method | Cmax mean | Cmax std | valid ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda item: (item["size"], item["method"])):
        lines.append(
            f"| {row['size']} | {row['method']} | {_fmt(row['Cmax_roll_mean'])} | "
            f"{_fmt(row['Cmax_roll_std'])} | {_fmt(row.get('valid_ratio', 0.0), 2)} |"
        )

    lines.extend(["", "## Best Heuristic By Size", "", "| size | best method | Cmax mean |", "|---|---:|---:|"])
    for size, row in sorted(best_by_size.items()):
        lines.append(f"| {size} | {row['method']} | {_fmt(row['Cmax_roll_mean'])} |")

    lines.extend(
        [
            "",
            "## Split Effect Summary",
            "",
            "| size | ordering | split strategy | Cmax mean | valid ratio | cmax check pass ratio |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(split_summary_rows, key=lambda item: (item["size"], item["ordering"], item["split_strategy"])):
        lines.append(
            f"| {row['size']} | {row['ordering']} | {row['split_strategy']} | "
            f"{_fmt(row['Cmax_roll_mean'])} | {_fmt(row.get('valid_ratio', 0.0), 2)} | "
            f"{_fmt(row.get('cmax_check_pass_ratio', 0.0), 2)} |"
        )

    lines.extend(
        [
            "",
            "## Schedule Validation",
            "",
            f"- Baseline groups with invalid schedules: {len(invalid_baseline)}",
            f"- Split-effect groups with invalid schedules: {len(invalid_split)}",
            f"- Any invalid schedule detected: {'yes' if invalid_baseline or invalid_split else 'no'}",
            "",
            "## Next Stage Recommendation",
            "",
            "Stage A has fixed datasets, unified evaluation, split-effect diagnostics, and schedule validation in place. "
            "The next step is Stage B: build LookaheadGreedy, BeamSearch, and HybridTopK candidate sets for later candidate-set experiments.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=["small", "medium", "large"])
    args = parser.parse_args()
    run_stage_a(args.sizes)
    print(f"Stage A results saved under {RESULT_DIR}")


if __name__ == "__main__":
    main()
