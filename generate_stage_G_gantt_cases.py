"""Generate representative Stage G Gantt case comparisons."""

from __future__ import annotations

import argparse
import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List


BASELINE_DIR = Path("data/results/stage_G/baseline_eval/runs")
OUTPUT_DIR = Path("data/results/stage_G/gantt_cases")


def token() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def collect_detail(root: Path) -> List[dict]:
    if not root.exists():
        print(f"Warning: baseline detail root does not exist: {root}")
        return []
    rows: List[dict] = []
    for path in sorted(root.glob("*/baseline_eval_detail__*.csv")):
        rows.extend(read_csv(path))
    return rows


def fnum(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def select_cases(rows: List[dict], top_k: int) -> List[dict]:
    grouped: Dict[tuple, Dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault((row["scenario_run_id"], row["instance_id"]), {})[row["method"]] = row
    cases = []
    for (run_id, instance_id), methods in grouped.items():
        if "HGCR-PPO" not in methods or "FIFO" not in methods:
            continue
        gap = fnum(methods["FIFO"]["Cmax"]) - fnum(methods["HGCR-PPO"]["Cmax"])
        if gap > 0:
            cases.append({"scenario_run_id": run_id, "instance_id": instance_id, "gap": gap, "methods": methods})
    return sorted(cases, key=lambda item: item["gap"], reverse=True)[:top_k]


def run(args):
    rows = collect_detail(Path(args.baseline_dir))
    cases = select_cases(rows, args.top_k)
    out_dir = Path(args.output_dir) / token()
    print(f"Candidate cases: {len(cases)}")
    for case in cases:
        print(f"  - {case['scenario_run_id']} / {case['instance_id']} gap={case['gap']:.3f}")
    print(f"Planned output dir: {out_dir}")
    if args.dry_run:
        print("Dry run enabled: no Gantt figures will be written.")
        return out_dir
    if args.no_write:
        print("No-write enabled: selected cases without writing figures.")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.txt").write_text(
        "This placeholder records selected Gantt cases. Reconstructing exact schedules requires replaying the source Stage G run policies.\n"
        "Use evaluate_stage_G_dynamic_baselines.py outputs to identify scenario_run_id and instance_id.\n",
        encoding="utf-8",
    )
    print("Warning: reusable Gantt case selection is implemented; exact schedule replay hook is pending for full visual rendering.")
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
