"""Build a lightweight Stage F summary table without changing prior outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from utils.experiment_io import write_csv


DEFAULT_METHODS = [
    "FIFO",
    "Oracle Debug",
    "MLP-Ranker soft_ce",
    "RuleSelectorPPO",
    "RankerFallbackPPO",
    "DeltaRulePPO",
]
OUTPUT_FIELDS = [
    "method",
    "size",
    "split",
    "top_k",
    "Cmax_mean",
    "Cmax_std",
    "ranker_baseline_mean",
    "delta_improvement_mean",
    "delta_improvement_std",
    "source_file",
]


def collect_delta_rule_rows(result_dir: Path) -> List[Dict]:
    rows = []
    for path in sorted((result_dir / "delta_rule_ppo").glob("delta_rule_ppo_eval_summary*.csv")):
        with path.open("r", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "method": row.get("method", "DeltaRulePPO"),
                        "size": row.get("size", ""),
                        "split": row.get("split", ""),
                        "top_k": row.get("top_k", ""),
                        "Cmax_mean": row.get("Cmax_mean", ""),
                        "Cmax_std": row.get("Cmax_std", ""),
                        "ranker_baseline_mean": row.get("ranker_baseline_mean", ""),
                        "delta_improvement_mean": row.get("delta_improvement_mean", ""),
                        "delta_improvement_std": row.get("delta_improvement_std", ""),
                        "source_file": str(path),
                    }
                )
    return rows


def collect_rule_selector_rows(result_dir: Path) -> List[Dict]:
    rows = []
    for path in sorted(result_dir.glob("rule_selector_ppo_eval_summary_clean.csv")):
        with path.open("r", newline="") as f:
            for row in csv.DictReader(f):
                method = row.get("method", "")
                if method == "rule_selector_ppo":
                    method = "RuleSelectorPPO"
                rows.append(
                    {
                        "method": method,
                        "size": row.get("size", ""),
                        "split": row.get("split", ""),
                        "top_k": row.get("top_k", ""),
                        "Cmax_mean": row.get("Cmax_roll_mean", ""),
                        "Cmax_std": row.get("Cmax_roll_std", ""),
                        "ranker_baseline_mean": "",
                        "delta_improvement_mean": "",
                        "delta_improvement_std": "",
                        "source_file": str(path),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="data/results/stage_F")
    parser.add_argument("--output", default="data/results/stage_F/stage_F_summary.csv")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    rows = [*collect_rule_selector_rows(result_dir), *collect_delta_rule_rows(result_dir)]
    write_csv(rows, args.output, OUTPUT_FIELDS)
    print(f"Saved Stage F summary to {args.output} with methods target list: {', '.join(DEFAULT_METHODS)}")


if __name__ == "__main__":
    main()

