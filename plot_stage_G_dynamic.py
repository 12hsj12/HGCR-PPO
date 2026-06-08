"""Generate paper-ready Stage G dynamic rolling HGCR-PPO figures.

The script reads the latest Stage G summary CSVs and writes both PNG and PDF
figures. It never reads Stage E/F summaries by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence


SUMMARY_DIR = Path("data/results/stage_G/summary")
OUTPUT_DIR = Path("data/results/stage_G/figures")
ARRIVAL_ORDER = ["low", "medium", "high"]
BETA_ORDER = [0.01, 1.0, 5.0]
RULE_ORDER = ["FIFO", "GreedyECT", "Lookahead", "MLP-Ranker"]
COLORS = {
    "HGCR-PPO": "#2F6BFF",
    "FIFO": "#4A4A4A",
    "MLP-Ranker": "#D14B3F",
    "GreedyECT": "#4EAD5B",
    "Lookahead": "#A06CD5",
}


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
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def canonical_method(value: str) -> str:
    if value in {"HGCR-PPO", "HGCR_Dynamic_PPO", "HGCR-Dynamic-PPO", "HGCR_PPO"}:
        return "HGCR-PPO"
    if value in {"MLPRanker", "MLP_Ranker", "MLP_Ranker_soft_ce", "MLP-Ranker"}:
        return "MLP-Ranker"
    return value


def get_value(row: dict, *names: str, default=0.0) -> float:
    for name in names:
        if name in row and row[name] not in {"", None}:
            return fnum(row[name], default)
    return default


def hgcr_cmax(row: dict) -> float:
    return get_value(row, "HGCR_PPO_Cmax_mean", "HGCR-PPO", "HGCR_PPO", "Cmax_mean")


def mlp_cmax(row: dict) -> float:
    return get_value(row, "MLPRanker_Cmax_mean", "MLP-Ranker", "MLPRanker", "MLP_Ranker_soft_ce")


def fifo_cmax(row: dict) -> float:
    return get_value(row, "FIFO_Cmax_mean", "FIFO")


def rel_fifo(row: dict) -> float:
    explicit = get_value(row, "relative_to_FIFO", default=float("nan"))
    if not math.isnan(explicit):
        return explicit
    fifo = fifo_cmax(row)
    return (fifo - hgcr_cmax(row)) / max(fifo, 1e-8)


def rel_mlp(row: dict) -> float:
    explicit = get_value(row, "relative_to_MLPRanker", "relative_to_MLP-Ranker", default=float("nan"))
    if not math.isnan(explicit):
        return explicit
    mlp = mlp_cmax(row)
    return (mlp - hgcr_cmax(row)) / max(mlp, 1e-8)


def ordered_beta_rows(rows: Sequence[dict]) -> List[dict]:
    by_beta = {round(fnum(row.get("reward_beta")), 6): row for row in rows}
    ordered = [by_beta[round(beta, 6)] for beta in BETA_ORDER if round(beta, 6) in by_beta]
    extras = [row for row in rows if round(fnum(row.get("reward_beta")), 6) not in {round(beta, 6) for beta in BETA_ORDER}]
    return ordered + sorted(extras, key=lambda row: fnum(row.get("reward_beta")))


def ordered_seed_rows(rows: Sequence[dict]) -> List[dict]:
    return sorted(rows, key=lambda row: int(row.get("seed", 999)) if str(row.get("seed", "")).isdigit() else 999)


def ordered_arrival_rows(rows: Sequence[dict]) -> List[dict]:
    order = {name: idx for idx, name in enumerate(ARRIVAL_ORDER)}
    return sorted(rows, key=lambda row: order.get(row.get("arrival_intensity"), 99))


def mpl():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )
    return plt


def save_figure(fig, out_dir: Path, stem: str, no_write: bool) -> List[Path]:
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    print(f"Plan plot: {paths[0]} and {paths[1]}")
    if no_write:
        return paths
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def zoom_ylim(ax, values: Iterable[float], pad: float) -> None:
    vals = [value for value in values if not math.isnan(value)]
    if not vals:
        return
    lo = min(vals) - pad
    hi = max(vals) + pad
    if hi <= lo:
        hi = lo + 1.0
    ax.set_ylim(lo, hi)


def annotate_bars(ax, bars, fmt: str = "{:.2f}") -> None:
    ymin, ymax = ax.get_ylim()
    span = max(1e-8, ymax - ymin)
    for bar in bars:
        height = bar.get_height()
        offset = 0.02 * span if height >= 0 else -0.06 * span
        va = "bottom" if height >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, height + offset, fmt.format(height), ha="center", va=va, fontsize=8)


def moving_average(values: Sequence[float], window: int) -> List[float]:
    window = max(1, int(window))
    out = []
    cumsum = 0.0
    queue: List[float] = []
    for value in values:
        queue.append(value)
        cumsum += value
        if len(queue) > window:
            cumsum -= queue.pop(0)
        out.append(cumsum / len(queue))
    return out


def best_so_far(values: Sequence[float]) -> List[float]:
    out = []
    best = float("inf")
    for value in values:
        best = min(best, value)
        out.append(best)
    return out


def beta_rows(data: Dict[str, List[dict]]) -> List[dict]:
    return ordered_beta_rows(data["beta"])


def seed_rows(data: Dict[str, List[dict]]) -> List[dict]:
    return ordered_seed_rows(data["seed"])


def arrival_rows(data: Dict[str, List[dict]]) -> List[dict]:
    return ordered_arrival_rows(data["arrival"])


def action_rows(data: Dict[str, List[dict]], predicate: Callable[[dict], bool]) -> List[dict]:
    return [row for row in data["actions"] if predicate(row)]


def plot_beta_cmax_line(data: Dict[str, List[dict]], out_dir: Path, no_write: bool, _window: int) -> List[Path]:
    rows = beta_rows(data)
    if not rows:
        print("Warning: missing beta ablation rows; skip stage_G_beta_cmax_line")
        return []
    plt = mpl()
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    x = [fnum(row.get("reward_beta")) for row in rows]
    hgcr = [hgcr_cmax(row) for row in rows]
    fifo = [fifo_cmax(row) for row in rows]
    mlp = [mlp_cmax(row) for row in rows]
    ax.plot(x, hgcr, marker="o", color=COLORS["HGCR-PPO"], linewidth=1.8, label="HGCR-PPO")
    ax.axhline(fifo[0], color=COLORS["FIFO"], linestyle="--", linewidth=1.2, label="FIFO")
    ax.axhline(mlp[0], color=COLORS["MLP-Ranker"], linestyle=":", linewidth=1.4, label="MLP-Ranker")
    for beta, value in zip(x, hgcr):
        if abs(beta - 5.0) < 1e-9:
            ax.annotate(f"{value:.2f}", xy=(beta, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("Reward beta")
    ax.set_ylabel("Cmax")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    zoom_ylim(ax, [*hgcr, *fifo, *mlp], 2.0)
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "stage_G_beta_cmax_line", no_write)
    plt.close(fig)
    return paths


def plot_beta_improvement_bar(data: Dict[str, List[dict]], out_dir: Path, no_write: bool, _window: int) -> List[Path]:
    rows = beta_rows(data)
    if not rows:
        print("Warning: missing beta ablation rows; skip stage_G_beta_improvement_bar")
        return []
    plt = mpl()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    labels = [str(fnum(row.get("reward_beta"))) for row in rows]
    values = [rel_fifo(row) * 100.0 for row in rows]
    bars = ax.bar(labels, values, color=COLORS["HGCR-PPO"], width=0.58)
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.set_xlabel("Reward beta")
    ax.set_ylabel("Improvement over FIFO (%)")
    annotate_bars(ax, bars, "{:.2f}%")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "stage_G_beta_improvement_bar", no_write)
    plt.close(fig)
    return paths


def plot_seed_improvement_bar(data: Dict[str, List[dict]], out_dir: Path, no_write: bool, _window: int) -> List[Path]:
    rows = seed_rows(data)
    if not rows:
        print("Warning: missing seed stability rows; skip stage_G_seed_improvement_bar")
        return []
    plt = mpl()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    labels = [f"seed{row.get('seed')}" for row in rows]
    values = [rel_fifo(row) * 100.0 for row in rows]
    bars = ax.bar(labels, values, color=COLORS["HGCR-PPO"], width=0.58)
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.set_ylabel("Improvement over FIFO (%)")
    annotate_bars(ax, bars, "{:.2f}%")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "stage_G_seed_improvement_bar", no_write)
    plt.close(fig)
    return paths


def plot_arrival_improvement_dual(data: Dict[str, List[dict]], out_dir: Path, no_write: bool, _window: int) -> List[Path]:
    rows = arrival_rows(data)
    if not rows:
        print("Warning: missing arrival generalization rows; skip stage_G_arrival_improvement_dual")
        return []
    plt = mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    labels = [row["arrival_intensity"] for row in rows]
    series = [
        ("(a) over FIFO", [rel_fifo(row) * 100.0 for row in rows]),
        ("(b) over MLP-Ranker", [rel_mlp(row) * 100.0 for row in rows]),
    ]
    for ax, (title, values) in zip(axes, series):
        bars = ax.bar(labels, values, color=COLORS["HGCR-PPO"], width=0.58)
        ax.axhline(0.0, color="#222222", linewidth=0.9)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Improvement (%)")
        annotate_bars(ax, bars, "{:.2f}%")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(w_pad=2.0)
    paths = save_figure(fig, out_dir, "stage_G_arrival_improvement_dual", no_write)
    plt.close(fig)
    return paths


def plot_arrival_cmax_grouped_zoom(data: Dict[str, List[dict]], out_dir: Path, no_write: bool, _window: int) -> List[Path]:
    rows = arrival_rows(data)
    if not rows:
        print("Warning: missing arrival generalization rows; skip stage_G_arrival_cmax_grouped_zoom")
        return []
    import numpy as np

    plt = mpl()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    labels = [row["arrival_intensity"] for row in rows]
    x = np.arange(len(labels))
    width = 0.24
    values = {
        "FIFO": [fifo_cmax(row) for row in rows],
        "MLP-Ranker": [mlp_cmax(row) for row in rows],
        "HGCR-PPO": [hgcr_cmax(row) for row in rows],
    }
    for idx, name in enumerate(["FIFO", "MLP-Ranker", "HGCR-PPO"]):
        bars = ax.bar(x + (idx - 1) * width, values[name], width=width, label=name, color=COLORS[name])
        annotate_bars(ax, bars, "{:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cmax")
    zoom_ylim(ax, [value for vals in values.values() for value in vals], 5.0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "stage_G_arrival_cmax_grouped_zoom", no_write)
    plt.close(fig)
    return paths


def group_action_ratios(rows: Sequence[dict], label_fn: Callable[[dict], str], label_order: Sequence[str]) -> tuple[List[str], Dict[str, List[float]]]:
    grouped: Dict[str, Dict[str, float]] = {}
    for row in rows:
        label = label_fn(row)
        grouped.setdefault(label, {rule: 0.0 for rule in RULE_ORDER})
        rule = row.get("rule_name", "")
        if rule in grouped[label]:
            grouped[label][rule] += fnum(row.get("selection_ratio"))
    labels = [label for label in label_order if label in grouped]
    labels.extend(label for label in grouped if label not in labels)
    values = {rule: [grouped[label].get(rule, 0.0) for label in labels] for rule in RULE_ORDER}
    return labels, values


def stacked_action_plot(
    rows: Sequence[dict],
    label_fn: Callable[[dict], str],
    label_order: Sequence[str],
    stem: str,
    out_dir: Path,
    no_write: bool,
) -> List[Path]:
    if not rows:
        print(f"Warning: missing action ratio rows; skip {stem}")
        return []
    import numpy as np

    labels, values = group_action_ratios(rows, label_fn, label_order)
    if not labels:
        print(f"Warning: no grouped action ratios; skip {stem}")
        return []
    plt = mpl()
    fig, ax = plt.subplots(figsize=(max(3.8, 0.85 * len(labels)), 2.8))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for rule in RULE_ORDER:
        vals = np.asarray(values[rule])
        ax.bar(x, vals, bottom=bottom, label=rule, color=COLORS.get(rule, None), width=0.62)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Selection ratio")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=4)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, stem, no_write)
    plt.close(fig)
    return paths


def plot_action_ratio_by_beta(data: Dict[str, List[dict]], out_dir: Path, no_write: bool, _window: int) -> List[Path]:
    rows = action_rows(
        data,
        lambda row: row.get("arrival_intensity") == "medium"
        and row.get("carryover_ratio") == "medium"
        and str(row.get("seed")) == "0"
        and round(fnum(row.get("reward_beta")), 6) in {round(beta, 6) for beta in BETA_ORDER},
    )
    return stacked_action_plot(
        rows,
        lambda row: f"b={fnum(row.get('reward_beta')):g}",
        [f"b={beta:g}" for beta in BETA_ORDER],
        "stage_G_action_ratio_by_beta",
        out_dir,
        no_write,
    )


def plot_action_ratio_by_seed(data: Dict[str, List[dict]], out_dir: Path, no_write: bool, _window: int) -> List[Path]:
    rows = action_rows(
        data,
        lambda row: row.get("arrival_intensity") == "medium"
        and row.get("carryover_ratio") == "medium"
        and round(fnum(row.get("reward_beta")), 6) == 5.0
        and str(row.get("seed")) in {"0", "1", "2"},
    )
    return stacked_action_plot(
        rows,
        lambda row: f"seed{row.get('seed')}",
        ["seed0", "seed1", "seed2"],
        "stage_G_action_ratio_by_seed",
        out_dir,
        no_write,
    )


def plot_action_ratio_by_arrival(data: Dict[str, List[dict]], out_dir: Path, no_write: bool, _window: int) -> List[Path]:
    rows = action_rows(
        data,
        lambda row: round(fnum(row.get("reward_beta")), 6) == 5.0
        and str(row.get("seed")) == "0"
        and row.get("carryover_ratio") == "medium"
        and row.get("arrival_intensity") in set(ARRIVAL_ORDER),
    )
    return stacked_action_plot(
        rows,
        lambda row: row.get("arrival_intensity", ""),
        ARRIVAL_ORDER,
        "stage_G_action_ratio_by_arrival",
        out_dir,
        no_write,
    )


def find_training_curve(data: Dict[str, List[dict]]) -> tuple[dict | None, List[dict]]:
    candidates = [
        row
        for row in data["all_runs"]
        if canonical_method(row.get("method", "")) == "HGCR-PPO"
        and row.get("arrival_intensity") == "medium"
        and row.get("carryover_ratio") == "medium"
        and str(row.get("seed")) == "0"
        and round(fnum(row.get("reward_beta")), 6) == 5.0
        and row.get("reward_curve_path")
    ]
    if not candidates:
        return None, []
    path = Path(candidates[0]["reward_curve_path"])
    rows = read_csv(path)
    if not rows:
        run_dir = Path(candidates[0].get("source_run_dir", ""))
        rows = read_csv(run_dir / "train_log.csv")
    return candidates[0], rows


def plot_training_reward_ma(data: Dict[str, List[dict]], out_dir: Path, no_write: bool, window: int) -> List[Path]:
    _, rows = find_training_curve(data)
    if not rows:
        print("Warning: missing beta=5.0 medium seed0 reward curve; skip stage_G_training_reward_ma")
        return []
    plt = mpl()
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    episodes = [int(float(row["episode"])) for row in rows]
    reward = [get_value(row, "episode_reward", "total_reward") for row in rows]
    ma = moving_average(reward, window)
    ax.plot(episodes, reward, color=COLORS["HGCR-PPO"], alpha=0.10, linewidth=0.6, label="raw")
    ax.plot(episodes, ma, color=COLORS["HGCR-PPO"], linewidth=1.8, label=f"MA-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "stage_G_training_reward_ma", no_write)
    plt.close(fig)
    return paths


def plot_training_cmax_ma(data: Dict[str, List[dict]], out_dir: Path, no_write: bool, window: int) -> List[Path]:
    summary_row, rows = find_training_curve(data)
    if not rows:
        print("Warning: missing beta=5.0 medium seed0 Cmax curve; skip stage_G_training_cmax_ma")
        return []
    plt = mpl()
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    episodes = [int(float(row["episode"])) for row in rows]
    cmax = [get_value(row, "episode_Cmax", "Cmax_ppo", "final_cmax") for row in rows]
    ma = moving_average(cmax, window)
    best = best_so_far(cmax)
    ax.plot(episodes, ma, color=COLORS["HGCR-PPO"], linewidth=1.8, label=f"Cmax MA-{window}")
    ax.plot(episodes, best, color="#111111", linewidth=1.1, linestyle="--", label="best-so-far")
    if summary_row is not None:
        ax.axhline(fifo_cmax(summary_row), color=COLORS["FIFO"], linewidth=1.1, linestyle=":", label="FIFO")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Cmax")
    zoom_ylim(ax, [*ma, *best, fifo_cmax(summary_row or {})], 3.0)
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "stage_G_training_cmax_ma", no_write)
    plt.close(fig)
    return paths


def plot_paper_composite(data: Dict[str, List[dict]], out_dir: Path, no_write: bool, _window: int) -> List[Path]:
    beta = beta_rows(data)
    seed = seed_rows(data)
    arrival = arrival_rows(data)
    actions = action_rows(
        data,
        lambda row: round(fnum(row.get("reward_beta")), 6) == 5.0
        and str(row.get("seed")) == "0"
        and row.get("carryover_ratio") == "medium"
        and row.get("arrival_intensity") in set(ARRIVAL_ORDER),
    )
    if not beta or not seed or not arrival or not actions:
        print("Warning: incomplete composite inputs; skip stage_G_paper_composite")
        return []
    import numpy as np

    plt = mpl()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.5))
    ax = axes[0][0]
    labels = [str(fnum(row.get("reward_beta"))) for row in beta]
    bars = ax.bar(labels, [rel_fifo(row) * 100.0 for row in beta], color=COLORS["HGCR-PPO"], width=0.58)
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.set_title("(a) Reward scaling", loc="left")
    ax.set_ylabel("Improvement over FIFO (%)")
    annotate_bars(ax, bars, "{:.2f}%")
    ax.grid(axis="y", alpha=0.22)

    ax = axes[0][1]
    labels = [f"seed{row.get('seed')}" for row in seed]
    bars = ax.bar(labels, [rel_fifo(row) * 100.0 for row in seed], color=COLORS["HGCR-PPO"], width=0.58)
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.set_title("(b) Seed stability", loc="left")
    ax.set_ylabel("Improvement over FIFO (%)")
    annotate_bars(ax, bars, "{:.2f}%")
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1][0]
    labels = [row["arrival_intensity"] for row in arrival]
    bars = ax.bar(labels, [rel_fifo(row) * 100.0 for row in arrival], color=COLORS["HGCR-PPO"], width=0.58)
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.set_title("(c) Arrival generalization", loc="left")
    ax.set_ylabel("Improvement over FIFO (%)")
    annotate_bars(ax, bars, "{:.2f}%")
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1][1]
    act_labels, act_values = group_action_ratios(actions, lambda row: row.get("arrival_intensity", ""), ARRIVAL_ORDER)
    x = np.arange(len(act_labels))
    bottom = np.zeros(len(act_labels))
    for rule in RULE_ORDER:
        vals = np.asarray(act_values[rule])
        ax.bar(x, vals, bottom=bottom, label=rule, color=COLORS.get(rule, None), width=0.62)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(act_labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("(d) Rule selection", loc="left")
    ax.set_ylabel("Selection ratio")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2)
    ax.grid(axis="y", alpha=0.20)
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    paths = save_figure(fig, out_dir, "stage_G_paper_composite", no_write)
    plt.close(fig)
    return paths


PLOTTERS: List[Callable[[Dict[str, List[dict]], Path, bool, int], List[Path]]] = [
    plot_beta_cmax_line,
    plot_beta_improvement_bar,
    plot_seed_improvement_bar,
    plot_arrival_improvement_dual,
    plot_arrival_cmax_grouped_zoom,
    plot_action_ratio_by_beta,
    plot_action_ratio_by_seed,
    plot_action_ratio_by_arrival,
    plot_training_reward_ma,
    plot_training_cmax_ma,
    plot_paper_composite,
]


def run(args) -> List[Path]:
    summary_dir = Path(args.summary_dir)
    out_dir = Path(args.output_dir) if args.output_dir else output_subdir(OUTPUT_DIR)
    inputs = discover(summary_dir)
    print("Stage G summary inputs:")
    for key, path in inputs.items():
        print(f"  - {key}: {path if path else 'missing'}")
    print(f"Planned figure dir: {out_dir}")
    if args.dry_run:
        print("Dry run enabled: no figures will be written.")
    no_write = args.no_write or args.dry_run
    if not no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
    data = {key: read_csv(path) for key, path in inputs.items()}
    outputs: List[Path] = []
    for plotter in PLOTTERS:
        outputs.extend(plotter(data, out_dir, no_write, args.rolling_window))
    if args.no_write:
        print("No-write enabled: plot inputs checked without writing PNG/PDF files.")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", default=str(SUMMARY_DIR))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--rolling_window", type=int, default=100)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
