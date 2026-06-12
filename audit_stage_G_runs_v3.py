"""Audit Stage G paper-v3 training coverage and emit only missing commands."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


RUNS_DIR = Path("data/results/stage_G/hgcr_dynamic_ppo/runs")
DEFAULT_BAT = Path("data/results/stage_G/summary/missing_runs_paper_v3.bat")
DEFAULT_CSV = Path("data/results/stage_G/summary/missing_runs_paper_v3.csv")


@dataclass(frozen=True)
class RunKey:
    size: str
    arrival_intensity: str
    carryover_ratio: str
    reward_beta: float
    seed: int
    episodes: int = 5000
    top_k: int = 5
    reward_mode: str = "util_plus_cmax"
    baseline_method: str = "fifo"
    eval_interval: int = 50


@dataclass
class Target:
    key: RunKey
    groups: set[str] = field(default_factory=set)


def fnum(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def add_target(targets: Dict[RunKey, Target], key: RunKey, group: str) -> None:
    targets.setdefault(key, Target(key)).groups.add(group)


def paper_v3_targets() -> List[Target]:
    targets: Dict[RunKey, Target] = {}
    for size in ["small", "medium", "large"]:
        for seed in [0, 1, 2]:
            add_target(targets, RunKey(size, "medium", "medium", 5.0, seed), "A_main_convergence")
    for beta in [0.01, 0.1, 1.0, 2.0, 5.0]:
        add_target(targets, RunKey("small", "medium", "medium", beta, 0), "B_beta_sensitivity")
    for arrival in ["low", "medium", "high"]:
        for seed in [0, 1, 2]:
            add_target(targets, RunKey("small", arrival, "medium", 5.0, seed), "C_arrival_multiseed")
    for arrival in ["low", "medium", "high"]:
        for carryover in ["low", "medium", "high"]:
            add_target(targets, RunKey("small", arrival, carryover, 5.0, 0), "D_heatmap_3x3")
    return sorted(targets.values(), key=lambda item: (item.key.size, item.key.arrival_intensity, item.key.carryover_ratio, item.key.reward_beta, item.key.seed))


def read_manifest(run_dir: Path) -> tuple[dict, str | None]:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}, "manifest_missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError):
        return {}, "manifest_invalid"


def first_file(run_dir: Path, pattern: str) -> Path | None:
    paths = sorted(run_dir.glob(pattern))
    return paths[0] if paths else None


def max_episode(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    maximum = 0
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                maximum = max(maximum, int(fnum(row.get("episode"))))
    except OSError:
        return 0
    return maximum


def manifest_value(manifest: dict, name: str, default=None):
    if name in manifest:
        return manifest[name]
    return (manifest.get("args") or {}).get(name, default)


def matches(manifest: dict, key: RunKey) -> bool:
    return (
        manifest_value(manifest, "size") == key.size
        and manifest_value(manifest, "arrival_intensity") == key.arrival_intensity
        and manifest_value(manifest, "carryover_ratio") == key.carryover_ratio
        and round(fnum(manifest_value(manifest, "reward_beta")), 6) == round(key.reward_beta, 6)
        and int(fnum(manifest_value(manifest, "seed"), -1)) == key.seed
        and int(fnum(manifest_value(manifest, "top_k"), -1)) == key.top_k
        and manifest_value(manifest, "reward_mode") == key.reward_mode
        and manifest_value(manifest, "baseline_method") == key.baseline_method
    )


def validate_run(run_dir: Path, manifest: dict, key: RunKey) -> List[str]:
    reasons = []
    episodes = int(fnum(manifest_value(manifest, "episodes")))
    completed = int(fnum(manifest.get("completed_episodes", episodes)))
    eval_interval = int(fnum(manifest_value(manifest, "eval_interval"), 10**9))
    if manifest.get("failed") is True:
        reasons.append("manifest_failed")
    if episodes < key.episodes:
        reasons.append(f"episodes_lt_{key.episodes}")
    if completed < key.episodes:
        reasons.append(f"completed_episodes_lt_{key.episodes}")
    if not bool(manifest_value(manifest, "disable_early_stop", False)) and bool(manifest.get("early_stopped", False)):
        reasons.append("early_stopped")
    if eval_interval > key.eval_interval:
        reasons.append(f"eval_interval_gt_{key.eval_interval}")

    eval_path = first_file(run_dir, "eval_history*.csv")
    action_history = first_file(run_dir, "action_history*.csv")
    action_stage = first_file(run_dir, "action_stage_summary*.csv")
    if eval_path is None:
        reasons.append("eval_history_missing")
    elif max_episode(eval_path) < key.episodes:
        reasons.append(f"eval_history_max_episode_lt_{key.episodes}")
    if action_history is None:
        reasons.append("action_history_missing")
    if action_stage is None:
        reasons.append("action_stage_summary_missing")
    return reasons


def scan_runs(runs_dir: Path) -> List[tuple[Path, dict, str | None]]:
    if not runs_dir.exists():
        return []
    scanned = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        manifest, error = read_manifest(run_dir)
        scanned.append((run_dir, manifest, error))
    return scanned


def command(key: RunKey) -> str:
    return (
        "python run_hgcr_dynamic_ppo.py"
        f" --size {key.size} --top_k {key.top_k} --episodes {key.episodes} --seed {key.seed}"
        f" --arrival_intensity {key.arrival_intensity} --carryover_ratio {key.carryover_ratio}"
        f" --reward_mode {key.reward_mode} --reward_beta {key.reward_beta:g}"
        f" --baseline_method {key.baseline_method} --disable_early_stop --eval_interval {key.eval_interval}"
    )


def audit(targets: Iterable[Target], scanned) -> List[dict]:
    rows = []
    for target in targets:
        candidates = []
        for run_dir, manifest, manifest_error in scanned:
            if manifest_error or not matches(manifest, target.key):
                continue
            reasons = validate_run(run_dir, manifest, target.key)
            candidates.append((run_dir, manifest, reasons))
        valid = [item for item in candidates if not item[2]]
        if valid:
            chosen = max(valid, key=lambda item: item[0].stat().st_mtime)
            status = "valid"
            reasons = ""
            run_dir = str(chosen[0])
        elif candidates:
            chosen = max(candidates, key=lambda item: item[0].stat().st_mtime)
            status = "invalid"
            reasons = ";".join(chosen[2])
            run_dir = str(chosen[0])
        else:
            status = "missing"
            reasons = "no_matching_run"
            run_dir = ""
        key = target.key
        rows.append(
            {
                "profile_groups": "+".join(sorted(target.groups)),
                "size": key.size,
                "arrival_intensity": key.arrival_intensity,
                "carryover_ratio": key.carryover_ratio,
                "reward_beta": key.reward_beta,
                "seed": key.seed,
                "episodes": key.episodes,
                "top_k": key.top_k,
                "reward_mode": key.reward_mode,
                "baseline_method": key.baseline_method,
                "required_eval_interval": key.eval_interval,
                "status": status,
                "reasons": reasons,
                "matched_run_dir": run_dir,
                "missing_command": "" if status == "valid" else command(key),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    fields = [
        "profile_groups", "size", "arrival_intensity", "carryover_ratio", "reward_beta", "seed",
        "episodes", "top_k", "reward_mode", "baseline_method", "required_eval_interval", "status",
        "reasons", "matched_run_dir", "missing_command",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    if args.profile != "paper_v3":
        raise ValueError(f"Unsupported profile: {args.profile}")
    targets = paper_v3_targets()
    scanned = scan_runs(Path(args.runs_dir))
    rows = audit(targets, scanned)
    missing = [row for row in rows if row["status"] != "valid"]
    valid = [row for row in rows if row["status"] == "valid"]
    output_rows = missing if args.exclude_existing_valid else rows
    print(f"Scanned run directories: {len(scanned)}")
    print(f"Unique paper_v3 targets: {len(rows)}")
    print(f"Valid targets: {len(valid)}")
    print(f"Missing or invalid targets: {len(missing)}")
    print(f"Planned CSV: {args.output_csv}")
    print(f"Planned BAT: {args.output_bat}")
    for row in missing:
        print(f"  [{row['status']}] {row['missing_command']} ({row['reasons']})")
    if args.dry_run:
        print("Dry run enabled: audit outputs will not be written.")
        return rows
    write_csv(Path(args.output_csv), output_rows)
    if args.write_missing_commands:
        bat_path = Path(args.output_bat)
        bat_path.parent.mkdir(parents=True, exist_ok=True)
        bat_path.write_text("\r\n".join(row["missing_command"] for row in missing) + ("\r\n" if missing else ""), encoding="utf-8")
    print("Stage G paper_v3 run audit saved.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default=str(RUNS_DIR))
    parser.add_argument("--profile", default="paper_v3")
    parser.add_argument("--write_missing_commands", action="store_true")
    parser.add_argument("--output_bat", default=str(DEFAULT_BAT))
    parser.add_argument("--output_csv", default=str(DEFAULT_CSV))
    parser.add_argument("--exclude_existing_valid", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
