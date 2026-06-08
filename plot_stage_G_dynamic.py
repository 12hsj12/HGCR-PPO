"""Plot clean Stage G dynamic rolling HGCR-PPO summaries."""

from __future__ import annotations

import argparse
import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List


SUMMARY_DIR = Path("data/results/stage_G/summary")
OUTPUT_DIR = Path("data/results/stage_G/figures")
RULE_ORDER = ["FIFO", "GreedyECT", "Lookahead", "MLP-Ranker"]
ARRIVAL_ORDER = ["low", "medium", "high"]


def output_subdir(base: Path) -> Path:
    return base / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def latest(summary_dir: Path, pattern: str) -> Path | None:
    matches = sorted(summary_dir.glob(pattern), key=lambda path: path.stat().st_mtime if path.exists() else 0.0)
    return matches[-1] if matches else None


def discover(summary_dir: Path) -> Dict[str, Path | None]:
    return {
        "all_runs": latest(summary_dir, "stage_G_all_runs__*.csv"),
        "beta": latest(summary_dir, "stage_G_beta_ablation__*.csv"),
        "seed": latest(summary_dir, "stage_G_seed_stability__*.csv"),
        "arrival": latest(summary_dir, "stage_G_arrival_generalization__*.csv"),
        "actions": latest(summary_dir, "stage_G_action_ratio_summary__*.csv"),
    }


def read_csv(path: Path | None) -> List[dict]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
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
    width = min(0.34, 0.82 / max(1, len(names)))
    start = -width * (len(names) - 1) / 2
    for idx, name in enumerate(names):
        plt.bar(x + start + idx * width, series[name], width, label=name)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()


def plot_beta(paths: Dict[str, Path | None], out_dir: Path, no_write: bool) -> Path | None:
    rows = sorted(read_csv(paths["beta"]), key=lambda row: fnum(row.get("reward_beta")))
    if not rows:
        print("Warning: missing beta ablation data; skip stage_G_beta_ablation_bar.png")
        return None
    out = out_dir / "stage_G_beta_ablation_bar.png"
    print(f"Plan plot: {out}")
    if no_write:
        return out
    plt = matplotlib()
    plt.figure(figsize=(7.2, 4.5))
    grouped_bar(
        plt,
        [str(row["reward_beta"]) for row in rows],
        {
            "FIFO": [fnum(row["FIFO_Cmax_mean"]) for row in rows],
            "HGCR-PPO": [fnum(row["HGCR_PPO_Cmax_mean"]) for row in rows],
        },
        "Cmax mean",
        "Stage G reward beta ablation",
    )
    plt.xlabel("reward_beta")
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_seed(paths: Dict[str, Path | None], out_dir: Path, no_write: bool) -> Path | None:
    rows = sorted(read_csv(paths["seed"]), key=lambda row: int(row["seed"]) if str(row.get("seed")).isdigit() else 999)
    if not rows:
        print("Warning: missing seed stability data; skip stage_G_seed_stability_bar.png")
        return None
    out = out_dir / "stage_G_seed_stability_bar.png"
    print(f"Plan plot: {out}")
    if no_write:
        return out
    plt = matplotlib()
    plt.figure(figsize=(7.2, 4.5))
    grouped_bar(
        plt,
        [f"seed{row['seed']}" for row in rows],
        {
            "FIFO": [fnum(row["FIFO_Cmax_mean"]) for row in rows],
            "HGCR-PPO": [fnum(row["HGCR_PPO_Cmax_mean"]) for row in rows],
        },
        "Cmax mean",
        "Stage G seed stability",
    )
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_arrival(paths: Dict[str, Path | None], out_dir: Path, no_write: bool) -> Path | None:
    order = {name: idx for idx, name in enumerate(ARRIVAL_ORDER)}
    rows = sorted(read_csv(paths["arrival"]), key=lambda row: order.get(row.get("arrival_intensity"), 99))
    if not rows:
        print("Warning: missing arrival generalization data; skip stage_G_arrival_generalization_bar.png")
        return None
    out = out_dir / "stage_G_arrival_generalization_bar.png"
    print(f"Plan plot: {out}")
    if no_write:
        return out
    plt = matplotlib()
    plt.figure(figsize=(7.2, 4.5))
    grouped_bar(
        plt,
        [row["arrival_intensity"] for row in rows],
        {
            "FIFO": [fnum(row["FIFO_Cmax_mean"]) for row in rows],
            "MLP-Ranker": [fnum(row["MLPRanker_Cmax_mean"]) for row in rows],
            "HGCR-PPO": [fnum(row["HGCR_PPO_Cmax_mean"]) for row in rows],
        },
        "Cmax mean",
        "Stage G arrival generalization",
    )
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_actions(paths: Dict[str, Path | None], out_dir: Path, no_write: bool) -> Path | None:
    rows = read_csv(paths["actions"])
    if not rows:
        print("Warning: missing action ratio data; skip stage_G_action_ratio_stacked_bar.png")
        return None
    grouped: Dict[str, Dict[str, float]] = {}
    labels = []
    for row in rows:
        key = f"{row.get('arrival_intensity')}-b{row.get('reward_beta')}-s{row.get('seed')}"
        if key not in grouped:
            grouped[key] = {rule: 0.0 for rule in RULE_ORDER}
            labels.append(key)
        rule = row.get("rule_name")
        if rule in grouped[key]:
            grouped[key][rule] += fnum(row.get("selection_ratio"))
    out = out_dir / "stage_G_action_ratio_stacked_bar.png"
    print(f"Plan plot: {out}")
    if no_write:
        return out
    import numpy as np

    plt = matplotlib()
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    plt.figure(figsize=(max(7.2, 0.8 * len(labels)), 4.8))
    for rule in RULE_ORDER:
        values = [grouped[label][rule] for label in labels]
        plt.bar(x, values, bottom=bottom, label=rule)
        bottom += np.asarray(values)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Selection ratio")
    plt.title("Stage G rule action ratios")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_training(paths: Dict[str, Path | None], out_dir: Path, no_write: bool) -> Path | None:
    rows = [
        row
        for row in read_csv(paths["all_runs"])
        if row.get("method") == "HGCR-PPO"
        and row.get("arrival_intensity") == "medium"
        and row.get("carryover_ratio") == "medium"
        and str(row.get("seed")) == "0"
        and fnum(row.get("reward_beta")) == 5.0
        and row.get("reward_curve_path")
    ]
    if not rows:
        print("Warning: missing beta=5.0 medium seed0 training curve path; skip stage_G_training_curve_beta5_medium_seed0.png")
        return None
    curve_path = Path(rows[0]["reward_curve_path"])
    curve = read_csv(curve_path)
    if not curve:
        print(f"Warning: training curve CSV missing or empty: {curve_path}")
        return None
    out = out_dir / "stage_G_training_curve_beta5_medium_seed0.png"
    print(f"Plan plot: {out}")
    if no_write:
        return out
    plt = matplotlib()
    episodes = [int(float(row["episode"])) for row in curve]
    rewards = [fnum(row.get("episode_reward", row.get("total_reward"))) for row in curve]
    cmax = [fnum(row.get("episode_Cmax")) for row in curve]
    fig, left = plt.subplots(figsize=(7.5, 4.6))
    right = left.twinx()
    left.plot(episodes, rewards, label="episode_reward", color="#2F6BFF")
    right.plot(episodes, cmax, label="episode_Cmax", color="#D14B3F")
    left.set_xlabel("Episode")
    left.set_ylabel("Episode reward")
    right.set_ylabel("Episode Cmax")
    lines = left.get_lines() + right.get_lines()
    left.legend(lines, [line.get_label() for line in lines], frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def run(args) -> List[Path]:
    summary_dir = Path(args.summary_dir)
    out_dir = Path(args.output_dir) if args.output_dir else output_subdir(OUTPUT_DIR)
    inputs = discover(summary_dir)
    print("Stage G summary inputs:")
    for key, path in inputs.items():
        print(f"  - {key}: {path if path else 'missing'}")
    print(f"Planned figure dir: {out_dir}")
    plotters: List[Callable[[Dict[str, Path | None], Path, bool], Path | None]] = [
        plot_beta,
        plot_seed,
        plot_arrival,
        plot_actions,
        plot_training,
    ]
    if args.max_plots is not None:
        plotters = plotters[: max(0, args.max_plots)]
    no_write = args.no_write or args.dry_run
    if args.dry_run:
        print("Dry run enabled: no figures will be written.")
    if not no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for plotter in plotters:
        path = plotter(inputs, out_dir, no_write)
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
