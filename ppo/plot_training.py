"""Training and comparison plots for PPO experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def _read_csv(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="") as f:
        return list(csv.DictReader(f))


def _float(rows, key):
    return [float(row[key]) if row.get(key, "") not in {"", "None"} else float("nan") for row in rows]


def _save_line(path: Path, x, series, title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for label, values in series.items():
        ax.plot(x, values, label=label)
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_eval_line(path: Path, x, train_values, eval_values, title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, train_values, label="train_final_Cmax")
    eval_x = []
    eval_y = []
    for episode, value in zip(x, eval_values):
        if value == value:
            eval_x.append(episode)
            eval_y.append(value)
    if eval_x:
        ax.plot(eval_x, eval_y, marker="o", label="eval_Cmax_mean")
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def create_training_plots(log_path: str, size: str, episodes: int) -> None:
    rows = _read_csv(log_path)
    x = [int(row["episode"]) for row in rows]
    base = Path("data/results/ppo/plots")
    _save_line(
        base / f"reward_curve_{size}_{episodes}.png",
        x,
        {
            "raw_episode_reward": _float(rows, "raw_episode_reward"),
            "scaled_episode_reward": _float(rows, "scaled_episode_reward"),
        },
        "PPO reward",
        "reward",
    )
    _save_line(
        base / f"reward_curve_raw_{size}_{episodes}.png",
        x,
        {"raw_episode_reward": _float(rows, "raw_episode_reward")},
        "PPO raw reward",
        "raw reward",
    )
    _save_line(
        base / f"reward_curve_scaled_{size}_{episodes}.png",
        x,
        {"scaled_episode_reward": _float(rows, "scaled_episode_reward")},
        "PPO scaled reward",
        "scaled reward",
    )
    _save_eval_line(
        base / f"cmax_curve_{size}_{episodes}.png",
        x,
        _float(rows, "train_final_Cmax"),
        _float(rows, "eval_Cmax_mean"),
        "PPO Cmax",
        "Cmax",
    )
    _save_line(
        base / f"loss_curve_{size}_{episodes}.png",
        x,
        {"policy_loss": _float(rows, "policy_loss"), "value_loss": _float(rows, "value_loss")},
        "PPO losses",
        "loss",
    )
    _save_line(
        base / f"entropy_curve_{size}_{episodes}.png",
        x,
        {
            "total_entropy": _float(rows, "total_entropy"),
            "job_entropy": _float(rows, "job_entropy"),
            "split_entropy": _float(rows, "split_entropy"),
        },
        "PPO entropy",
        "entropy",
    )
    _save_line(
        base / f"split_num_curve_{size}_{episodes}.png",
        x,
        {"average_selected_split_num": _float(rows, "average_selected_split_num")},
        "Selected split number",
        "average split_num",
    )
    _save_line(
        base / f"split_num_ratio_curve_{size}_{episodes}.png",
        x,
        {
            "split_num_1_ratio": _float(rows, "split_num_1_ratio"),
            "split_num_2_ratio": _float(rows, "split_num_2_ratio"),
            "split_num_3_ratio": _float(rows, "split_num_3_ratio"),
            "split_num_4_ratio": _float(rows, "split_num_4_ratio"),
        },
        "Selected split number ratios",
        "ratio",
    )
    _save_line(
        base / f"value_diagnostics_{size}_{episodes}.png",
        x,
        {
            "value_pred_mean": _float(rows, "value_pred_mean"),
            "value_target_mean": _float(rows, "value_target_mean"),
            "explained_variance": _float(rows, "explained_variance"),
        },
        "Value diagnostics",
        "value",
    )


def create_split_distribution_plot(counts: Dict[int, int], size: str, episodes: int) -> None:
    path = Path("data/results/ppo/plots") / f"split_num_distribution_{size}_{episodes}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = sorted(counts)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(x) for x in xs], [counts[x] for x in xs])
    ax.set_xlabel("split_num")
    ax.set_ylabel("frequency")
    ax.set_title("PPO split number distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def create_comparison_plots(size: str, episodes: int, ppo_metrics: Dict[str, float], heuristic_rows: List[Dict[str, float]]) -> None:
    methods = ["PPO"] + [row["method"] for row in heuristic_rows]
    comparisons = [
        ("cmax", "Cmax_roll", ppo_metrics["test_Cmax_mean"], "Cmax"),
        ("utilization", "machine_utilization", ppo_metrics["test_machine_utilization"], "machine utilization"),
        ("waiting_time", "average_waiting_time", ppo_metrics["test_average_waiting_time"], "average waiting time"),
    ]
    base = Path("data/results/ppo/plots")
    base.mkdir(parents=True, exist_ok=True)
    for suffix, key, ppo_value, ylabel in comparisons:
        values = [ppo_value] + [float(row[key]) for row in heuristic_rows]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(methods, values)
        ax.set_ylabel(ylabel)
        ax.set_title(f"PPO vs heuristics - {ylabel}")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(base / f"ppo_vs_heuristics_{suffix}_{size}_{episodes}.png", dpi=160)
        plt.close(fig)


def update_episode_sensitivity() -> None:
    result_dir = Path("data/results/ppo")
    rows = []
    for path in sorted(result_dir.glob("ppo_eval_*_*.csv")):
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        size = parts[2]
        episodes = int(parts[3])
        with path.open("r", newline="") as f:
            data = next(csv.DictReader(f), None)
        if data:
            rows.append({"size": size, "episodes": episodes, "test_Cmax_mean": data["test_Cmax_mean"]})
    if not rows:
        return
    summary = result_dir / "episode_sensitivity_summary.csv"
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["size", "episodes", "test_Cmax_mean"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["size"], r["episodes"])))

    for size in ["small", "medium", "large"]:
        size_rows = sorted([r for r in rows if r["size"] == size], key=lambda r: r["episodes"])
        if len(size_rows) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["episodes"] for r in size_rows], [float(r["test_Cmax_mean"]) for r in size_rows], marker="o")
        ax.set_xlabel("episodes")
        ax.set_ylabel("test_Cmax_mean")
        ax.set_title(f"Episode sensitivity - {size}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = Path("data/results/ppo/plots") / f"episode_sensitivity_cmax_{size}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
