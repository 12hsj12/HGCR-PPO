"""Build a non-overwriting Stage F summary from run manifests."""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from utils.experiment_io import write_csv


OUTPUT_FIELDS = [
    "run_id",
    "method",
    "checkpoint_type",
    "size",
    "split",
    "top_k",
    "episodes",
    "seed",
    "Cmax_mean",
    "Cmax_std",
    "ranker_baseline_mean",
    "delta_improvement_mean",
    "delta_improvement_std",
    "normalized_delta_improvement_mean",
    "keep_ranker_ratio_mean",
    "override_ratio_mean",
    "fallback_count_mean",
    "valid_schedule_rate",
    "source_file",
    "manifest_file",
]


def unique_summary_path(summary_dir: Path) -> Path:
    summary_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        path = summary_dir / f"stage_F_summary__{stamp}_{suffix}.csv"
        if not path.exists():
            return path
    raise RuntimeError("Could not create a unique stage_F_summary output path.")


def read_manifest(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_delta_rule_runs(result_dir: Path) -> List[Dict]:
    rows = []
    runs_dir = result_dir / "delta_rule_ppo" / "runs"
    manifest_paths = [*runs_dir.glob("*/mf__*.json"), *runs_dir.glob("*/manifest__*.json")]
    for manifest_path in sorted(manifest_paths):
        manifest = read_manifest(manifest_path)
        run_id = manifest.get("run_id", manifest_path.parent.name)
        summary_paths = [
            *manifest_path.parent.glob("s[lb]__*.csv"),
            *manifest_path.parent.glob("eval_summary_*__*.csv"),
        ]
        for summary_path in sorted(summary_paths):
            with summary_path.open("r", newline="") as f:
                for row in csv.DictReader(f):
                    normalized = {field: row.get(field, "") for field in OUTPUT_FIELDS}
                    normalized["run_id"] = normalized.get("run_id") or run_id
                    normalized["source_file"] = str(summary_path)
                    normalized["manifest_file"] = str(manifest_path)
                    rows.append(normalized)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="data/results/stage_F")
    parser.add_argument("--summary_dir", default="data/results/stage_F/summary")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    rows = collect_delta_rule_runs(result_dir)
    output_path = unique_summary_path(Path(args.summary_dir))
    write_csv(rows, output_path, OUTPUT_FIELDS)
    print(f"Saved non-overwriting Stage F summary to {output_path}")


if __name__ == "__main__":
    main()

