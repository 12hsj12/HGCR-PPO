"""Plot paper-level Stage G figures from paper_results tables."""

from __future__ import annotations

import argparse
import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List


INPUT_DIR = Path("data/results/stage_G/paper_results")
OUTPUT_DIR = Path("data/results/stage_G/paper_figures")
METHOD_ORDER = ["Random", "SPT", "LPT", "MinLoad", "GreedyECT", "Lookahead", "FIFO", "MLP-Ranker", "HGCR-PPO"]
ARRIVAL_ORDER = ["low", "medium", "high"]
RULES = ["FIFO", "GreedyECT", "Lookahead", "MLP-Ranker"]


def token() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime if path.exists() else 0.0)
    return matches[-1] if matches else None


def discover(root: Path) -> Dict[str, Path | None]:
    return {
        "detail": latest(root, "stage_G_method_comparison_detail__*.csv"),
        "summary": latest(root, "stage_G_method_comparison_summary__*.csv"),
        "wtl": latest(root, "stage_G_win_tie_loss__*.csv"),
        "rank": latest(root, "stage_G_rank_summary__*.csv"),
        "heatmap": latest(root, "stage_G_scenario_heatmap__*.csv"),
        "action": latest(root, "stage_G_action_performance_summary__*.csv"),
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


def mpl():
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
    return plt


def save(fig, out_dir: Path, name: str, no_write: bool):
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    print(f"Plan plot: {png} and {pdf}")
    if not no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")


def group_summary(rows: List[dict]) -> Dict[tuple, float]:
    grouped: Dict[tuple, List[float]] = {}
    for row in rows:
        grouped.setdefault((row.get("arrival_intensity"), row.get("method")), []).append(fnum(row.get("Cmax_mean")))
    return {key: sum(vals) / len(vals) for key, vals in grouped.items() if vals}


def fig_g1(data, out_dir, no_write):
    rows = data["summary"]
    if not rows:
        print("Warning: skip G1, missing summary.")
        return
    import numpy as np

    plt = mpl()
    values = group_summary(rows)
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    x = np.arange(len(ARRIVAL_ORDER))
    width = 0.085
    for idx, method in enumerate(METHOD_ORDER):
        y = [values.get((arrival, method), 0.0) for arrival in ARRIVAL_ORDER]
        ax.plot(x, y, marker="o", linewidth=1.2, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(ARRIVAL_ORDER)
    ax.set_ylabel("Cmax mean")
    ax.set_xlabel("Arrival intensity")
    ax.legend(frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.28), loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, out_dir, "fig_G1_multi_method_cmax_by_arrival", no_write)
    plt.close(fig)


def fig_g2(data, out_dir, no_write):
    rows = data["detail"]
    if not rows:
        print("Warning: skip G2, missing detail.")
        return
    plt = mpl()
    grouped = [[fnum(row["Cmax"]) for row in rows if row.get("method") == method] for method in METHOD_ORDER]
    labels = [method for method, vals in zip(METHOD_ORDER, grouped) if vals]
    values = [vals for vals in grouped if vals]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.set_ylabel("Per-instance Cmax")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, out_dir, "fig_G2_method_boxplot_cmax", no_write)
    plt.close(fig)


def fig_g3(data, out_dir, no_write):
    rows = data["wtl"]
    if not rows:
        print("Warning: skip G3, missing win/tie/loss.")
        return
    import numpy as np

    plt = mpl()
    rows = [row for row in rows if row.get("baseline_method") != "HGCR-PPO"]
    labels = [row["baseline_method"] for row in rows]
    win = np.array([fnum(row["win_rate"]) for row in rows])
    tie = np.array([fnum(row["tie_rate"]) for row in rows])
    loss = np.array([fnum(row["loss_rate"]) for row in rows])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh(y, win, label="Win", color="#2F6BFF")
    ax.barh(y, tie, left=win, label="Tie", color="#AAAAAA")
    ax.barh(y, loss, left=win + tie, label="Loss", color="#D14B3F")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Rate")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    save(fig, out_dir, "fig_G3_win_tie_loss", no_write)
    plt.close(fig)


def fig_g4(data, out_dir, no_write):
    rows = data["rank"]
    if not rows:
        print("Warning: skip G4, missing rank summary.")
        return
    plt = mpl()
    rows = sorted(rows, key=lambda row: fnum(row.get("mean_rank")))
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.bar([row["method"] for row in rows], [fnum(row["mean_rank"]) for row in rows], color="#2F6BFF")
    ax.set_ylabel("Mean rank (lower is better)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, out_dir, "fig_G4_rank_distribution", no_write)
    plt.close(fig)


def fig_g5(data, out_dir, no_write):
    rows = data["heatmap"]
    if not rows:
        print("Warning: skip G5, missing heatmap.")
        return
    carryovers = sorted({row["carryover_ratio"] for row in rows})
    if len(carryovers) < 2:
        print("Warning: only one carryover level found; heatmap will be a thin matrix.")
    import numpy as np

    plt = mpl()
    matrix = np.full((len(ARRIVAL_ORDER), len(carryovers)), np.nan)
    for row in rows:
        if row["arrival_intensity"] in ARRIVAL_ORDER:
            matrix[ARRIVAL_ORDER.index(row["arrival_intensity"]), carryovers.index(row["carryover_ratio"])] = fnum(row["HGCR_improvement_over_MLP"]) * 100.0
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(carryovers)))
    ax.set_xticklabels(carryovers)
    ax.set_yticks(range(len(ARRIVAL_ORDER)))
    ax.set_yticklabels(ARRIVAL_ORDER)
    ax.set_xlabel("Carryover ratio")
    ax.set_ylabel("Arrival intensity")
    fig.colorbar(im, ax=ax, label="Improvement over MLP-Ranker (%)")
    fig.tight_layout()
    save(fig, out_dir, "fig_G5_dynamic_scenario_heatmap", no_write)
    plt.close(fig)


def fig_g6(data, out_dir, no_write):
    rows = data["action"]
    if not rows:
        print("Warning: skip G6, missing action-performance summary.")
        return
    import numpy as np

    plt = mpl()
    rows = sorted(rows, key=lambda row: (row["arrival_intensity"], row["carryover_ratio"], str(row["seed"])))
    labels = [f"{row['arrival_intensity']}-{row['carryover_ratio']}-s{row['seed']}" for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(6.5, len(labels) * 0.75), 5.0), sharex=True)
    axes[0].bar(x, [fnum(row["HGCR_relative_to_MLP"]) * 100.0 for row in rows], color="#2F6BFF")
    axes[0].axhline(0.0, color="#222222", linewidth=0.8)
    axes[0].set_ylabel("Improvement over MLP (%)")
    bottom = np.zeros(len(labels))
    for rule in RULES:
        col = "MLPRanker_ratio" if rule == "MLP-Ranker" else f"{rule}_ratio"
        vals = np.array([fnum(row[col]) for row in rows])
        axes[1].bar(x, vals, bottom=bottom, label=rule)
        bottom += vals
    axes[1].set_ylabel("Action ratio")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    save(fig, out_dir, "fig_G6_action_performance_panel", no_write)
    plt.close(fig)


def fig_g7(data, out_dir, no_write):
    print("Warning: G7 training eval curve requires training curve CSV from HGCR runs; use plot_stage_G_dynamic.py for reward/Cmax MA curves.")


def fig_g8(data, out_dir, no_write):
    print("Warning: G8 reward scaling supplement is available in plot_stage_G_dynamic.py until paper_results stores beta curves.")


def run(args):
    inputs = discover(Path(args.input_dir))
    out_dir = Path(args.output_dir) / token()
    print("Paper result inputs:")
    for key, path in inputs.items():
        print(f"  - {key}: {path if path else 'missing'}")
    print(f"Planned figure dir: {out_dir}")
    if args.dry_run:
        print("Dry run enabled: no figures will be written.")
    no_write = args.no_write or args.dry_run
    if not no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
    data = {key: read_csv(path) for key, path in inputs.items()}
    for fn in [fig_g1, fig_g2, fig_g3, fig_g4, fig_g5, fig_g6, fig_g7, fig_g8]:
        fn(data, out_dir, no_write)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=str(INPUT_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
