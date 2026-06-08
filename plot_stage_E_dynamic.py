"""Plot Stage E dynamic rolling scheduling summary figures."""

from __future__ import annotations

import argparse
import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List


SUMMARY_DIR = Path("data/results/stage_E_dynamic_summary")
RULE_ORDER = ["FIFO", "GreedyECT", "Lookahead", "MLP-Ranker"]
ARRIVAL_ORDER = ["low", "medium", "high"]


def timestamp_uuid() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def latest_file(summary_dir: Path, pattern: str) -> Path | None:
    matches = sorted(summary_dir.glob(pattern), key=lambda path: path.stat().st_mtime if path.exists() else 0)
    return matches[-1] if matches else None


def read_csv(path: Path | None) -> List[dict]:
    if path is None:
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def require_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.5),
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    return plt


def grouped_bar(plt, labels, series: Dict[str, List[float]], ylabel: str, title: str) -> None:
    import numpy as np

    x = np.arange(len(labels))
    names = list(series)
    width = min(0.8 / max(1, len(names)), 0.35)
    offset = -width * (len(names) - 1) / 2
    for idx, name in enumerate(names):
        plt.bar(x + offset + idx * width, series[name], width, label=name)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()


def plot_beta_ablation(paths: Dict[str, Path | None], output_dir: Path, no_write: bool) -> Path | None:
    rows = read_csv(paths["beta"])
    if not rows:
        print("Warning: beta ablation CSV not found or empty; skipped beta_ablation_bar.png")
        return None
    rows = sorted(rows, key=lambda row: parse_float(row.get("reward_beta")))
    labels = [str(row["reward_beta"]) for row in rows]
    series = {
        "FIFO": [parse_float(row["FIFO_Cmax_mean"]) for row in rows],
        "HGCR-PPO": [parse_float(row["HGCR_PPO_Cmax_mean"]) for row in rows],
    }
    out = output_dir / "beta_ablation_bar.png"
    print(f"Plan plot: {out}")
    if no_write:
        return out
    plt = require_matplotlib()
    plt.figure()
    grouped_bar(plt, labels, series, "Cmax mean", "Reward beta sensitivity")
    plt.xlabel("reward_beta")
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_medium_seed(paths: Dict[str, Path | None], output_dir: Path, no_write: bool) -> Path | None:
    all_rows = read_csv(paths["all_runs"])
    rows = [
        row
        for row in all_rows
        if row.get("method") == "HGCR-PPO"
        and row.get("arrival_intensity") == "medium"
        and row.get("carryover_ratio") == "medium"
        and parse_float(row.get("reward_beta")) == 5.0
    ]
    if not rows:
        print("Warning: no medium-medium beta=5.0 seed rows; skipped medium_seed_comparison_bar.png")
        return None
    rows = sorted(rows, key=lambda row: str(row.get("seed")))
    labels = [f"seed{row['seed']}" for row in rows]
    series = {
        "FIFO": [parse_float(row["FIFO_Cmax_mean"]) for row in rows],
        "HGCR-PPO": [parse_float(row["Cmax_mean"]) for row in rows],
    }
    out = output_dir / "medium_seed_comparison_bar.png"
    print(f"Plan plot: {out}")
    if no_write:
        return out
    plt = require_matplotlib()
    plt.figure()
    grouped_bar(plt, labels, series, "Cmax mean", "Medium arrival / medium carryover stability")
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_arrival_generalization(paths: Dict[str, Path | None], output_dir: Path, no_write: bool) -> Path | None:
    rows = read_csv(paths["arrival"])
    if not rows:
        print("Warning: arrival generalization CSV not found or empty; skipped arrival_generalization_bar.png")
        return None
    order = {name: idx for idx, name in enumerate(ARRIVAL_ORDER)}
    rows = sorted(rows, key=lambda row: order.get(row.get("arrival_intensity"), 99))
    labels = [row["arrival_intensity"] for row in rows]
    series = {
        "FIFO": [parse_float(row["FIFO_Cmax_mean"]) for row in rows],
        "MLP-Ranker": [parse_float(row["MLPRanker_Cmax_mean"]) for row in rows],
        "HGCR-PPO": [parse_float(row["HGCR_PPO_Cmax_mean"]) for row in rows],
    }
    out = output_dir / "arrival_generalization_bar.png"
    print(f"Plan plot: {out}")
    if no_write:
        return out
    plt = require_matplotlib()
    plt.figure()
    grouped_bar(plt, labels, series, "Cmax mean", "Arrival intensity generalization")
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_action_ratio(paths: Dict[str, Path | None], output_dir: Path, no_write: bool) -> Path | None:
    rows = read_csv(paths["actions"])
    if not rows:
        print("Warning: action ratio CSV not found or empty; skipped action_ratio_stacked_bar.png")
        return None
    scenario_keys = []
    grouped: Dict[str, Dict[str, float]] = {}
    for row in rows:
        key = f"{row.get('arrival_intensity')}-b{row.get('reward_beta')}-s{row.get('seed')}"
        if key not in grouped:
            grouped[key] = {rule: 0.0 for rule in RULE_ORDER}
            scenario_keys.append(key)
        rule = row.get("rule_name")
        if rule in grouped[key]:
            grouped[key][rule] += parse_float(row.get("selection_ratio"))
    out = output_dir / "action_ratio_stacked_bar.png"
    print(f"Plan plot: {out}")
    if no_write:
        return out
    import numpy as np

    plt = require_matplotlib()
    x = np.arange(len(scenario_keys))
    bottoms = np.zeros(len(scenario_keys))
    plt.figure(figsize=(max(7.2, len(scenario_keys) * 0.8), 4.8))
    for rule in RULE_ORDER:
        values = [grouped[key][rule] for key in scenario_keys]
        plt.bar(x, values, bottom=bottoms, label=rule)
        bottoms += np.asarray(values)
    plt.xticks(x, scenario_keys, rotation=25, ha="right")
    plt.ylabel("Selection ratio")
    plt.title("Rule action ratios")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_training_curve(paths: Dict[str, Path | None], output_dir: Path, no_write: bool) -> Path | None:
    all_rows = read_csv(paths["all_runs"])
    candidates = [
        row
        for row in all_rows
        if row.get("method") == "HGCR-PPO"
        and row.get("arrival_intensity") == "medium"
        and row.get("carryover_ratio") == "medium"
        and str(row.get("seed")) == "0"
        and parse_float(row.get("reward_beta")) == 5.0
        and row.get("reward_curve_path")
    ]
    if not candidates:
        print("Warning: no reward curve path for beta=5.0 medium seed0; skipped training curve plot")
        return None
    curve_path = Path(candidates[0]["reward_curve_path"])
    curve_rows = read_csv(curve_path if curve_path.exists() else None)
    if not curve_rows:
        print(f"Warning: reward curve CSV missing or empty: {curve_path}")
        return None
    out = output_dir / "training_curve_beta5_medium_seed0.png"
    print(f"Plan plot: {out}")
    if no_write:
        return out
    plt = require_matplotlib()
    episodes = [int(float(row["episode"])) for row in curve_rows]
    rewards = [parse_float(row.get("episode_reward", row.get("total_reward"))) for row in curve_rows]
    cmax = [parse_float(row.get("episode_Cmax")) for row in curve_rows]
    fig, left = plt.subplots(figsize=(7.5, 4.6))
    right = left.twinx()
    left.plot(episodes, rewards, color="#2F6BFF", label="episode_reward")
    right.plot(episodes, cmax, color="#D14B3F", label="episode_Cmax")
    left.set_xlabel("Episode")
    left.set_ylabel("Episode reward")
    right.set_ylabel("Episode Cmax")
    lines = left.get_lines() + right.get_lines()
    left.legend(lines, [line.get_label() for line in lines], frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def discover_summary_files(summary_dir: Path) -> Dict[str, Path | None]:
    return {
        "all_runs": latest_file(summary_dir, "stage_E_all_runs__*.csv"),
        "beta": latest_file(summary_dir, "stage_E_beta_ablation__*.csv"),
        "arrival": latest_file(summary_dir, "stage_E_arrival_generalization__*.csv"),
        "actions": latest_file(summary_dir, "stage_E_action_ratio_summary__*.csv"),
    }


def run(args) -> List[Path]:
    summary_dir = Path(args.summary_dir)
    output_dir = Path(args.output_dir) if args.output_dir else summary_dir / "figures" / timestamp_uuid()
    paths = discover_summary_files(summary_dir)
    print("Summary inputs:")
    for name, path in paths.items():
        print(f"  - {name}: {path if path else 'missing'}")
    print(f"Planned figure dir: {output_dir}")
    plotters: List[Callable[[Dict[str, Path | None], Path, bool], Path | None]] = [
        plot_beta_ablation,
        plot_medium_seed,
        plot_arrival_generalization,
        plot_action_ratio,
        plot_training_curve,
    ]
    if args.max_plots is not None:
        plotters = plotters[: max(0, args.max_plots)]
    if args.dry_run:
        print("Dry run enabled: no figures will be written.")
    if not args.no_write and not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for plotter in plotters:
        path = plotter(paths, output_dir, no_write=args.no_write or args.dry_run)
        if path is not None:
            outputs.append(path)
    if args.no_write:
        print("No-write enabled: plot inputs checked without writing PNG files.")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", default=str(SUMMARY_DIR))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_plots", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
