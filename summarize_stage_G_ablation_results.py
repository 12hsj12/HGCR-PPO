"""Summarize Stage G ablation runs into no-FIFO, no-unknown paper assets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence


INPUT_DIR = Path("data/results/stage_G/ablation")
OUTPUT_DIR = INPUT_DIR / "summaries"
PAPER_DATA_DIR = Path("paper_zh/data_used")
DETAIL_FIELDS = [
    "ablation_family",
    "ablation_name",
    "size",
    "arrival_intensity",
    "carryover_ratio",
    "top_k",
    "episodes",
    "seed",
    "reward_mode",
    "reward_beta",
    "disabled_actions",
    "method",
    "Cmax_mean",
    "Cmax_std",
    "relative_to_MLPRanker",
    "valid_schedule_rate",
    "run_id",
    "manifest_path",
]
SUMMARY_FIELDS = [
    "ablation_family",
    "ablation_name",
    "size",
    "arrival_intensity",
    "carryover_ratio",
    "top_k",
    "episodes",
    "reward_mode",
    "reward_beta",
    "disabled_actions",
    "method",
    "seed_count",
    "Cmax_mean_across_seeds",
    "Cmax_std_across_seeds",
    "relative_to_MLPRanker_mean",
    "valid_schedule_rate_mean",
]


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def first_file(run_dir: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        path = run_dir / name
        if path.exists():
            return path
    return None


def load_manifest(run_dir: Path) -> dict | None:
    path = first_file(run_dir, ["manifest.json"])
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_detail_rows(input_dir: Path) -> List[dict]:
    rows: List[dict] = []
    runs_dir = input_dir / "runs"
    if not runs_dir.exists():
        return rows
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        manifest = load_manifest(run_dir)
        if not manifest or manifest.get("failed"):
            continue
        if manifest.get("stage") != "G" or manifest.get("experiment_family") != "hgcr_dynamic_ppo":
            continue
        family = manifest.get("ablation_family")
        if family not in {"reward_component", "action_library"}:
            continue
        if str(manifest.get("size", "")).lower() == "unknown":
            continue
        eval_path = first_file(run_dir, ["eval_summary.csv"])
        if eval_path is None:
            continue
        eval_rows = read_csv(eval_path)
        for row in eval_rows:
            if row.get("method") != "HGCR-PPO":
                continue
            rows.append(
                {
                    "ablation_family": family,
                    "ablation_name": manifest.get("ablation_name", ""),
                    "size": manifest.get("size", ""),
                    "arrival_intensity": manifest.get("arrival_intensity", ""),
                    "carryover_ratio": manifest.get("carryover_ratio", ""),
                    "top_k": manifest.get("top_k", ""),
                    "episodes": manifest.get("episodes", ""),
                    "seed": manifest.get("seed", ""),
                    "reward_mode": manifest.get("reward_mode", ""),
                    "reward_beta": manifest.get("reward_beta", ""),
                    "disabled_actions": "; ".join(manifest.get("disabled_actions") or []),
                    "method": row.get("method", ""),
                    "Cmax_mean": row.get("Cmax_mean", ""),
                    "Cmax_std": row.get("Cmax_std", ""),
                    "relative_to_MLPRanker": row.get("relative_to_MLPRanker", ""),
                    "valid_schedule_rate": row.get("valid_schedule_rate", ""),
                    "run_id": manifest.get("run_id", run_dir.name),
                    "manifest_path": str((run_dir / "manifest.json").as_posix()),
                }
            )
    return rows


def fnum(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize(rows: Sequence[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["ablation_family"],
            row["ablation_name"],
            row["size"],
            row["arrival_intensity"],
            row["carryover_ratio"],
            row["top_k"],
            row["episodes"],
            row["reward_mode"],
            row["reward_beta"],
            row["disabled_actions"],
            row["method"],
        )
        groups[key].append(row)
    out = []
    for key, vals in sorted(groups.items()):
        cmax = [fnum(row["Cmax_mean"]) for row in vals]
        rel = [fnum(row["relative_to_MLPRanker"]) for row in vals]
        valid = [fnum(row["valid_schedule_rate"]) for row in vals]
        out.append(
            {
                "ablation_family": key[0],
                "ablation_name": key[1],
                "size": key[2],
                "arrival_intensity": key[3],
                "carryover_ratio": key[4],
                "top_k": key[5],
                "episodes": key[6],
                "reward_mode": key[7],
                "reward_beta": key[8],
                "disabled_actions": key[9],
                "method": key[10],
                "seed_count": len(vals),
                "Cmax_mean_across_seeds": mean(cmax),
                "Cmax_std_across_seeds": pstdev(cmax) if len(cmax) > 1 else 0.0,
                "relative_to_MLPRanker_mean": mean(rel),
                "valid_schedule_rate_mean": mean(valid),
            }
        )
    return out


def write_family_outputs(rows: Sequence[dict], output_dir: Path, paper_data_dir: Path, export_paper: bool) -> List[Path]:
    written: List[Path] = []
    for family in ["reward_component", "action_library"]:
        family_rows = [row for row in rows if row["ablation_family"] == family]
        family_summary = summarize(family_rows)
        detail_name = f"stage_G_{family}_ablation_detail_no_fifo.csv"
        summary_name = f"stage_G_{family}_ablation_summary_no_fifo.csv"
        detail_path = output_dir / detail_name
        summary_path = output_dir / summary_name
        write_csv(detail_path, family_rows, DETAIL_FIELDS)
        write_csv(summary_path, family_summary, SUMMARY_FIELDS)
        written.extend([detail_path, summary_path])
        if export_paper:
            paper_detail = paper_data_dir / detail_name
            paper_summary = paper_data_dir / summary_name
            write_csv(paper_detail, family_rows, DETAIL_FIELDS)
            write_csv(paper_summary, family_summary, SUMMARY_FIELDS)
            written.extend([paper_detail, paper_summary])
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=str(INPUT_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--paper_data_dir", default=str(PAPER_DATA_DIR))
    parser.add_argument("--export_paper", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    rows = collect_detail_rows(Path(args.input_dir))
    print(f"Collected HGCR-PPO ablation rows: {len(rows)}")
    if args.dry_run:
        print("Dry run enabled: no summary files were written.")
        return
    written = write_family_outputs(rows, Path(args.output_dir), Path(args.paper_data_dir), args.export_paper)
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
