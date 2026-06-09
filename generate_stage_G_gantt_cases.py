"""Generate Stage G Gantt case comparisons from saved schedule traces."""

from __future__ import annotations

import argparse
import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List


BASELINE_DIR = Path("data/results/stage_G/baseline_eval/runs")
OUTPUT_DIR = Path("data/results/stage_G/gantt_cases")
METHODS = ["FIFO", "MLP-Ranker", "HGCR-PPO"]


def token() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def collect(root: Path, pattern: str) -> List[dict]:
    if not root.exists():
        print(f"Warning: root does not exist: {root}")
        return []
    rows: List[dict] = []
    for path in sorted(root.glob(f"*/{pattern}")):
        rows.extend(read_csv(path))
    return rows


def fnum(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def select_cases(detail: List[dict], top_k: int) -> List[dict]:
    grouped: Dict[tuple, Dict[str, dict]] = {}
    for row in detail:
        grouped.setdefault((row["scenario_run_id"], row["instance_id"]), {})[row["method"]] = row
    cases = []
    for (run_id, instance_id), methods in grouped.items():
        if "HGCR-PPO" not in methods or "FIFO" not in methods:
            continue
        gap = fnum(methods["FIFO"]["Cmax"]) - fnum(methods["HGCR-PPO"]["Cmax"])
        if gap > 0:
            cases.append({"scenario_run_id": run_id, "instance_id": instance_id, "gap": gap, "methods": methods})
    return sorted(cases, key=lambda item: item["gap"], reverse=True)[:top_k]


def trace_for_case(trace: List[dict], run_id: str, instance_id: str, method: str) -> List[dict]:
    return [row for row in trace if row["scenario_run_id"] == run_id and row["instance_id"] == instance_id and row["method"] == method]


def plot_method(ax, rows: List[dict], title: str) -> None:
    machines = sorted({row["machine_id"] for row in rows})
    machine_y = {machine: idx for idx, machine in enumerate(machines)}
    cmap = __import__("matplotlib.pyplot").pyplot.get_cmap("tab20")
    jobs = sorted({row["job_id"] for row in rows})
    colors = {job: cmap(idx % 20) for idx, job in enumerate(jobs)}
    for row in rows:
        start = fnum(row["start_time"])
        duration = fnum(row["duration"])
        ax.barh(machine_y[row["machine_id"]], duration, left=start, height=0.72, color=colors[row["job_id"]], edgecolor="black", linewidth=0.35)
    ax.set_yticks(list(machine_y.values()))
    ax.set_yticklabels(machines)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Time")
    ax.grid(axis="x", alpha=0.25)


def method_stats(case: dict, method: str) -> str:
    row = case["methods"].get(method)
    if not row:
        return method
    return f"{method}: Cmax={fnum(row['Cmax']):.1f}, util={fnum(row['machine_utilization']):.2f}, wait={fnum(row['average_waiting_time']):.1f}"


def save_fig(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def render_case(case: dict, trace: List[dict], out_dir: Path, idx: int) -> None:
    import matplotlib.pyplot as plt

    available = [method for method in METHODS if trace_for_case(trace, case["scenario_run_id"], case["instance_id"], method)]
    if not available:
        print(f"Warning: no trace rows for case {idx}.")
        return
    fig, axes = plt.subplots(len(available), 1, figsize=(11.0, max(3.0, 2.5 * len(available))), sharex=True)
    if len(available) == 1:
        axes = [axes]
    for ax, method in zip(axes, available):
        rows = trace_for_case(trace, case["scenario_run_id"], case["instance_id"], method)
        title = method_stats(case, method)
        if method == "HGCR-PPO":
            title += f", gap_vs_FIFO={case['gap']:.1f}"
        plot_method(ax, rows, title)
        save_fig(fig, out_dir, f"gantt_case{idx}_{method.replace('-', '_')}")
    fig.suptitle(f"Case {idx}: {case['scenario_run_id']} / {case['instance_id']}", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, f"gantt_case{idx}_comparison")
    plt.close(fig)


def run(args):
    detail = collect(Path(args.baseline_dir), "baseline_eval_detail__*.csv")
    trace = collect(Path(args.baseline_dir), "schedule_trace__*.csv")
    cases = select_cases(detail, args.top_k)
    out_dir = Path(args.output_dir) / token()
    print(f"Selected cases: {len(cases)}")
    for idx, case in enumerate(cases, 1):
        print(f"  - case{idx}: {case['scenario_run_id']} / {case['instance_id']} gap={case['gap']:.3f}")
    print(f"Planned output dir: {out_dir}")
    if args.dry_run:
        print("Dry run enabled: no Gantt figures will be written.")
        return out_dir
    if args.no_write:
        print("No-write enabled: selected cases without writing figures.")
        return out_dir
    if not trace:
        print("Warning: no schedule_trace CSV found. Re-run evaluate_stage_G_dynamic_baselines.py --save_schedule_trace.")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, case in enumerate(cases, 1):
        render_case(case, trace, out_dir, idx)
    print(f"Saved Gantt cases to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", default=str(BASELINE_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
