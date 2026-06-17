"""Create final Stage G no-FIFO paper figures from clean v3 statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence


PAPER_DIR = Path("data/results/stage_G/paper_results")
RUNS_DIR = Path("data/results/stage_G/hgcr_dynamic_ppo/runs")
OUTPUT_DIR = Path("data/results/stage_G/paper_figures")
SIZES = ["small", "medium", "large"]
ARRIVALS = ["low", "medium", "high"]
CARRYOVERS = ["low", "medium", "high"]
BETAS = [0.01, 0.1, 1.0, 2.0, 5.0]
METHODS = ["HGCR-PPO", "MLP-Ranker", "GreedyECT", "Lookahead", "MinLoad"]
MAIN_METHOD_ORDER = ["HGCR-PPO", "MLP-Ranker", "GreedyECT", "Lookahead", "MinLoad"]
DISTRIBUTION_METHOD_ORDER = ["HGCR-PPO", "MinLoad", "GreedyECT", "Lookahead", "MLP-Ranker"]
BASELINES = ["MLP-Ranker", "GreedyECT", "Lookahead", "MinLoad"]
ACTIONS = ["Arrival-order rule", "GreedyECT", "Lookahead", "MLP-Ranker"]
ACTION_LABELS = {
    "FIFO": "Arrival-order rule",
    "Arrival-order rule": "Arrival-order rule",
    "GreedyECT": "GreedyECT rule",
    "Lookahead": "Lookahead rule",
    "MLP-Ranker": "MLP-Ranker rule",
    "MLP_Ranker_soft_ce": "MLP-Ranker rule",
}
COLORS = {
    "HGCR-PPO": "#DB3124",
    "MLP-Ranker": "#4B74B2",
    "GreedyECT": "#FC8C5A",
    "Lookahead": "#90BEE0",
    "MinLoad": "#FFDF92",
    "Arrival-order rule": "#DB3124",
    "auxiliary": "#E6F1F3",
    "mean": "#111111",
}


def token() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}_no_fifo_v3_refined"


def latest(root: Path, pattern: str) -> Path | None:
    paths = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime if path.exists() else 0.0)
    return paths[-1] if paths else None


def read_csv(path: Path | None) -> List[dict]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str], no_write: bool) -> None:
    print(f"Plan CSV: {path}")
    if no_write:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) or math.isinf(number) else number


def moving_average(values: Sequence[float], window: int) -> List[float]:
    window = max(1, int(window))
    return [mean(values[max(0, idx - window + 1) : idx + 1]) for idx in range(len(values))]


def mpl():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    return plt


def save(fig, out_dir: Path, stem: str, no_write: bool) -> None:
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    print(f"Plan figure: {png} and {pdf}")
    if not no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")


def warn(name: str, message: str) -> None:
    print(f"Warning: {name}: {message}")


def validated_methods(methods: Sequence[str], *, include_hgcr: bool = False) -> List[str]:
    allowed = set(METHODS if include_hgcr else BASELINES)
    invalid = [method for method in methods if method not in allowed]
    if invalid:
        raise ValueError(f"Unsupported or excluded external methods: {invalid}. Allowed: {sorted(allowed)}")
    return list(dict.fromkeys(methods))


def clean_external(rows: Sequence[dict]) -> List[dict]:
    dropped_fifo = sum(1 for row in rows if row.get("method") == "FIFO")
    dropped_unknown = sum(1 for row in rows if row.get("size") not in {*SIZES, "all"})
    if dropped_fifo:
        print(f"Dropped FIFO rows from external plotting: n={dropped_fifo}")
    if dropped_unknown:
        print(f"Dropped unknown size rows from external plotting: n={dropped_unknown}")
    return [row for row in rows if row.get("method") != "FIFO" and row.get("size") in {*SIZES, "all"}]


def clean_significance(rows: Sequence[dict]) -> List[dict]:
    dropped = sum(1 for row in rows if row.get("baseline_method") == "FIFO")
    if dropped:
        print(f"Dropped FIFO rows from external plotting: n={dropped}")
    return [row for row in rows if row.get("baseline_method") != "FIFO"]


def read_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_run_tables(runs_dir: Path, min_required_episode: int) -> tuple[List[dict], List[dict]]:
    selected: Dict[tuple, tuple[bool, float, Path, dict, List[dict], int, int]] = {}
    if not runs_dir.exists():
        return [], []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        manifest = read_manifest(run_dir)
        size = manifest.get("size")
        if size not in SIZES:
            continue
        eval_path = next(iter(sorted(run_dir.glob("eval_history*.csv"))), None)
        eval_values = [row for row in read_csv(eval_path) if row.get("episode") not in {None, ""}]
        max_eval_episode = max((int(fnum(row.get("episode"))) for row in eval_values), default=0)
        configured_episodes = int(fnum(manifest.get("episodes") or (manifest.get("args") or {}).get("episodes"), min_required_episode))
        required_episode = max(int(min_required_episode), configured_episodes)
        disable_early_stop = manifest.get("disable_early_stop", (manifest.get("args") or {}).get("disable_early_stop"))
        disable_ok = disable_early_stop is True if disable_early_stop is not None else not bool(manifest.get("early_stopped", False))
        complete = bool(eval_values) and manifest.get("failed") is not True and disable_ok and max_eval_episode >= required_episode
        key = (
            size,
            manifest.get("arrival_intensity"),
            manifest.get("carryover_ratio"),
            round(fnum(manifest.get("reward_beta")), 6),
            str(manifest.get("seed")),
        )
        stamp = run_dir.stat().st_mtime
        candidate = (complete, stamp, run_dir, manifest, eval_values, max_eval_episode, required_episode)
        if key not in selected or (complete, stamp) > (selected[key][0], selected[key][1]):
            selected[key] = candidate

    eval_rows: List[dict] = []
    action_rows: List[dict] = []
    for complete, _, run_dir, manifest, eval_values, max_eval_episode, required_episode in selected.values():
        action_path = next(iter(sorted(run_dir.glob("action_stage_summary*.csv"))), None)
        for row in eval_values:
            row["size"] = row.get("size") or manifest.get("size")
            row["run_id"] = manifest.get("run_id", run_dir.name)
            row["run_complete"] = complete
            row["run_max_episode"] = max_eval_episode
            row["run_required_episode"] = required_episode
            eval_rows.append(row)
        for row in read_csv(action_path):
            row["size"] = row.get("size") or manifest.get("size")
            row["run_id"] = manifest.get("run_id", run_dir.name)
            row["action_name"] = "Arrival-order rule" if row.get("action_name") == "FIFO" else row.get("action_name")
            action_rows.append(row)
    return eval_rows, action_rows


def discover(args) -> Dict[str, object]:
    root = Path(args.paper_dir)
    files = {
        "summary": latest(root, "stage_G_method_comparison_summary_v3_no_fifo__*.csv"),
        "arpd": latest(root, "stage_G_arpd_summary_v3_no_fifo__*.csv"),
        "significance": latest(root, "stage_G_significance_tests_v3_no_fifo__*.csv"),
        "case": latest(root, "stage_G_case_curve_detail_v3_no_fifo__*.csv"),
        "scale": latest(root, "stage_G_scale_summary_v3_no_fifo__*.csv"),
        "audit": latest(root, "stage_G_data_audit_v3_no_fifo__*.csv"),
        "mapping": latest(root, "stage_G_case_mapping_v3_no_fifo__*.csv"),
    }
    missing = [name for name, path in files.items() if name in {"summary", "arpd", "significance", "case", "scale"} and path is None]
    data = {}
    for name, path in files.items():
        if name in {"audit", "mapping"}:
            continue
        rows = read_csv(path)
        data[name] = clean_significance(rows) if name == "significance" else clean_external(rows)
    data["audit"] = read_csv(files["audit"])
    data["mapping"] = read_csv(files["mapping"])
    eval_rows, action_rows = load_run_tables(Path(args.runs_dir), args.min_required_episode)
    data["eval_history"] = eval_rows
    data["action_stage"] = action_rows
    data["files"] = files
    data["missing_v3"] = missing
    return data


def filter_history(rows: Sequence[dict], *, size=None, beta=None, seed=None) -> List[dict]:
    out = []
    for row in rows:
        if row.get("arrival_intensity") != "medium" or row.get("carryover_ratio") != "medium":
            continue
        if size is not None and row.get("size") != size:
            continue
        if beta is not None and round(fnum(row.get("reward_beta")), 6) != round(beta, 6):
            continue
        if seed is not None and str(row.get("seed")) != str(seed):
            continue
        out.append(row)
    return sorted(out, key=lambda row: fnum(row.get("episode")))


def convergence_seed_rows(rows: Sequence[dict], size: str, seed: int) -> List[dict]:
    values = filter_history(rows, size=size, beta=5.0, seed=seed)
    return sorted(values, key=lambda row: int(fnum(row.get("episode"))))


def audit_convergence_size(rows: Sequence[dict], size: str, min_required_episode: int) -> Dict[int, dict]:
    print(f"Convergence audit for size={size}:")
    audit = {}
    for seed in [0, 1, 2]:
        values = convergence_seed_rows(rows, size, seed)
        max_episode = max((int(fnum(row.get("episode"))) for row in values), default=0)
        complete = bool(values) and all(bool(row.get("run_complete")) for row in values) and max_episode >= min_required_episode
        audit[seed] = {"rows": values, "max_episode": max_episode, "complete": complete}
        print(f"seed={seed} max_episode={max_episode} rows={len(values)} complete={complete}")
        if not complete:
            print(f"Warning: size={size} seed={seed} is incomplete and will be excluded from the mean curve by default.")
    return audit


def outer_episode_mean(seed_groups: Sequence[List[dict]], metric: str) -> tuple[List[int], List[float]]:
    values_by_episode: Dict[int, List[float]] = {}
    for rows in seed_groups:
        for row in rows:
            episode = int(fnum(row.get("episode")))
            if episode <= 0 or row.get(metric) in {None, ""}:
                continue
            values_by_episode.setdefault(episode, []).append(fnum(row.get(metric)))
    episodes = sorted(values_by_episode)
    return episodes, [mean(values_by_episode[episode]) for episode in episodes]


def plot_training_metric_by_size(data, out_dir, no_write, args, metric, stem_prefix, ylabel, baseline=False):
    rows = data["eval_history"]
    plt = mpl()
    for size in args.convergence_sizes:
        if size not in SIZES:
            print(f"Warning: unsupported convergence size={size}; skipped.")
            continue
        audit = audit_convergence_size(rows, size, args.min_required_episode)
        complete_groups = [entry["rows"] for entry in audit.values() if entry["complete"]]
        available_groups = [entry["rows"] for entry in audit.values() if entry["rows"]]
        exclude_incomplete = args.require_complete_convergence_runs or args.exclude_incomplete_from_mean
        mean_groups = complete_groups if exclude_incomplete else available_groups
        if len(complete_groups) < 3:
            print(f"Warning: size={size} has only {len(complete_groups)} complete seeds for the mean curve.")
        if not mean_groups:
            warn(stem_prefix, f"size={size} has no eligible seed data for mean aggregation")
            continue

        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        if args.show_raw_curves:
            for seed, entry in audit.items():
                if not entry["rows"]:
                    continue
                if not entry["complete"] and not args.allow_incomplete_raw_curves:
                    continue
                x = [int(fnum(row.get("episode"))) for row in entry["rows"]]
                y = [fnum(row.get(metric)) for row in entry["rows"]]
                ax.plot(
                    x,
                    y,
                    linewidth=0.65,
                    alpha=0.18 if entry["complete"] else 0.11,
                    color=COLORS["HGCR-PPO"],
                    linestyle="-" if entry["complete"] else "--",
                    label=f"seed{seed} raw" + ("" if entry["complete"] else " (incomplete)"),
                )

        mean_episodes, mean_values = outer_episode_mean(mean_groups, metric)
        smoothed = moving_average(mean_values, args.smoothing_window)
        if mean_episodes:
            ax.plot(mean_episodes, smoothed, linewidth=2.6, color=COLORS["mean"], label="Mean smoothed", zorder=5)
        mean_max = max(mean_episodes, default=0)
        print(f"mean_curve_max_episode={mean_max}")
        if mean_max < args.min_required_episode:
            max_by_seed = {seed: entry["max_episode"] for seed, entry in audit.items()}
            print(
                f"Warning: size={size} mean curve stops at episode {mean_max}, below required {args.min_required_episode}; "
                f"seed max episodes={max_by_seed}, eligible complete seeds={len(complete_groups)}."
            )

        if baseline:
            baseline_rows = [row for group in mean_groups for row in group]
            mlp = [fnum(row.get("baseline_MLPRanker_Cmax")) for row in baseline_rows if row.get("baseline_MLPRanker_Cmax") not in {None, ""}]
            if mlp:
                ax.axhline(mean(mlp), linestyle="--", linewidth=0.95, alpha=0.5, color=COLORS["MLP-Ranker"], label="MLP-Ranker baseline")
        plotted_values = [fnum(row.get(metric)) for group in available_groups for row in group if row.get(metric) not in {None, ""}]
        if metric == "eval_Cmax_mean" and plotted_values:
            y_min, y_max = min(plotted_values), max(plotted_values)
            margin = max(1.0, (y_max - y_min) * 0.12)
            ax.set_ylim(y_min - margin, y_max + margin)
        if metric == "eval_Cmax_mean" and mean_episodes:
            start_mean = mean_values[0]
            final_mean = mean_values[-1]
            best_mean = min(mean_values)
            ax.annotate(f"start {start_mean:.1f}", xy=(mean_episodes[0], smoothed[0]), xytext=(8, 10), textcoords="offset points", fontsize=7, color=COLORS["mean"])
            ax.annotate(f"best {best_mean:.1f}\nfinal {final_mean:.1f}", xy=(mean_episodes[-1], smoothed[-1]), xytext=(-72, 16), textcoords="offset points", fontsize=7, color=COLORS["mean"])
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="best")
        fig.tight_layout()
        save(fig, out_dir, f"{stem_prefix}_{size}_no_fifo", no_write)
        plt.close(fig)


def plot_fig2(data, out_dir, no_write, args):
    if not args.split_convergence_by_size:
        print("Warning: combined convergence figures are disabled; producing separate size files.")
    plot_training_metric_by_size(data, out_dir, no_write, args, "eval_Cmax_mean", "Fig_2_training_convergence", "Evaluation Cmax", baseline=True)
    plot_training_metric_by_size(data, out_dir, no_write, args, "eval_reward_mean", "Fig_2b_training_reward_convergence", "Evaluation reward")


def plot_fig3(data, out_dir, no_write, window, show_raw):
    rows = data["eval_history"]
    available = {round(fnum(row.get("reward_beta")), 6) for row in filter_history(rows, size="small", seed=0)}
    missing = [beta for beta in BETAS if round(beta, 6) not in available]
    if missing:
        print(f"Missing Fig_3 beta = {missing}")
    if len(missing) == len(BETAS):
        return
    plt = mpl()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    palette = [COLORS["HGCR-PPO"], COLORS["MLP-Ranker"], COLORS["GreedyECT"], COLORS["Lookahead"], COLORS["MinLoad"]]
    for beta, color in zip(BETAS, palette):
        values = filter_history(rows, size="small", beta=beta, seed=0)
        if not values:
            continue
        x = [fnum(row["episode"]) for row in values]
        for ax, metric in zip(axes, ["eval_Cmax_mean", "eval_reward_mean"]):
            y = [fnum(row[metric]) for row in values]
            if show_raw:
                ax.plot(x, y, color=color, alpha=0.16, linewidth=0.65)
            ax.plot(x, moving_average(y, window), color=color, linewidth=1.7, label=rf"$\beta$={beta:g}")
    axes[0].set_title("(a) Eval Cmax under different beta")
    axes[1].set_title("(b) Eval reward under different beta")
    axes[0].set_ylabel("Eval Cmax mean")
    axes[1].set_ylabel("Eval reward mean")
    for ax in axes:
        ax.set_xlabel("Episode")
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, ncol=1, loc="best")
    fig.tight_layout()
    save(fig, out_dir, "Fig_3_beta_sensitivity_training_curves_no_fifo", no_write)
    plt.close(fig)


def action_ratios(rows: Sequence[dict], size: str, start: int, end: int) -> Dict[str, float]:
    selected = [
        row
        for row in rows
        if row.get("size") == size
        and row.get("arrival_intensity") == "medium"
        and row.get("carryover_ratio") == "medium"
        and round(fnum(row.get("reward_beta")), 6) == 5.0
        and int(fnum(row.get("stage_start_episode"))) >= start
        and int(fnum(row.get("stage_end_episode"))) <= end
    ]
    result = {}
    for action in ACTIONS:
        values = [fnum(row.get("action_ratio")) for row in selected if row.get("action_name") == action]
        result[action] = mean(values) if values else 0.0
    return result


def plot_fig4(data, out_dir, no_write):
    rows = data["action_stage"]
    if not rows:
        warn("Fig_4", "missing action_stage_summary")
        return
    plt = mpl()
    import numpy as np

    stages = [(1, 1000), (1001, 2000), (2001, 3000), (3001, 4000), (4001, 5000)]
    labels = ["0-1000", "1000-2000", "2000-3000", "3000-4000", "4000-5000"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3), sharey=True)
    for ax, size in zip(axes, SIZES):
        x = np.arange(len(stages))
        bottom = np.zeros(len(stages))
        for action in ACTIONS:
            y = [action_ratios(rows, size, start, end)[action] for start, end in stages]
            ax.bar(x, y, bottom=bottom, color=COLORS.get(action), label=ACTION_LABELS[action])
            bottom += np.asarray(y)
        if not any(bottom):
            print(f"Missing Fig_4 action data: size={size}")
        ax.set_title(size)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Action percentage")
    fig.suptitle("HGCR-PPO action selection evolution", y=1.02)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, out_dir, "Fig_4_action_ratio_evolution_no_fifo_label", no_write)
    plt.close(fig)


def sampled_case_labels(rows: Sequence[dict], limit: int) -> List[str]:
    by_size = {size: [] for size in SIZES}
    for row in rows:
        label = row.get("case_label")
        if row.get("size") in by_size and label and label not in by_size[row["size"]]:
            by_size[row["size"]].append(label)
    for size in SIZES:
        by_size[size].sort(key=lambda label: int(label.lstrip("C")))
    quota = max(1, limit // len(SIZES))
    selected = []
    for size in SIZES:
        values = by_size[size]
        if len(values) <= quota:
            selected.extend(values)
        else:
            indexes = [round(idx * (len(values) - 1) / (quota - 1)) for idx in range(quota)] if quota > 1 else [0]
            selected.extend(values[idx] for idx in indexes)
    return selected[:limit]


def plot_fig5(data, out_dir, no_write, max_cases, compare_baselines):
    rows = data["case"]
    if not rows:
        warn("Fig_5", "missing v3 case detail")
        return
    labels = sampled_case_labels(rows, max_cases)
    mapping = []
    for label in labels:
        row = next(row for row in rows if row.get("case_label") == label)
        mapping.append({key: row.get(key, "") for key in ["case_label", "case_id", "size", "arrival_intensity", "carryover_ratio", "seed", "instance_id"]})
    write_csv(out_dir / "Fig_5_case_mapping_no_fifo.csv", mapping, ["case_label", "case_id", "size", "arrival_intensity", "carryover_ratio", "seed", "instance_id"], no_write)
    plt = mpl()
    fig, axes = plt.subplots(2, 1, figsize=(max(9.5, len(labels) * 0.17), 6.4), sharex=True)
    x = list(range(len(labels)))
    displayed_methods = [method for method in MAIN_METHOD_ORDER if method == "HGCR-PPO" or method in compare_baselines]
    for method in displayed_methods:
        y = []
        for label in labels:
            hit = next((row for row in rows if row.get("case_label") == label and row.get("method") == method), None)
            y.append(fnum(hit.get("Cmax")) if hit else float("nan"))
        axes[0].plot(x, y, marker=".", linewidth=1.0, label=method, color=COLORS[method])
    fields = {
        "MLP-Ranker": "improvement_vs_MLP_Ranker",
        "GreedyECT": "improvement_vs_GreedyECT",
        "Lookahead": "improvement_vs_Lookahead",
        "MinLoad": "improvement_vs_MinLoad",
    }
    for baseline in compare_baselines:
        field = fields[baseline]
        y = []
        for label in labels:
            hit = next((row for row in rows if row.get("case_label") == label and row.get("method") == "HGCR-PPO"), None)
            y.append(fnum(hit.get(field)) if hit and hit.get(field) != "" else float("nan"))
        axes[1].plot(x, y, linewidth=1.3, label=f"HGCR-PPO vs {baseline}", color=COLORS[baseline])

    previous_size = None
    for idx, label in enumerate(labels):
        size = next((row.get("size") for row in mapping if row.get("case_label") == label), None)
        if previous_size is not None and size != previous_size:
            for ax in axes:
                ax.axvline(idx - 0.5, color="#999999", linewidth=0.8, alpha=0.6)
        previous_size = size
    for size in SIZES:
        indexes = [idx for idx, row in enumerate(mapping) if row.get("size") == size]
        if indexes:
            axes[0].axvspan(min(indexes) - 0.5, max(indexes) + 0.5, alpha=0.045, color=COLORS["auxiliary"])
            axes[1].axvspan(min(indexes) - 0.5, max(indexes) + 0.5, alpha=0.035, color=COLORS["auxiliary"])
    axes[0].set_title("(a) Cmax across representative cases")
    axes[1].set_title("(b) Relative improvement of HGCR-PPO over baselines")
    axes[0].set_ylabel("Cmax")
    axes[1].set_ylabel("Improvement (%)")
    step = max(1, len(labels) // 12)
    axes[1].set_xticks(x[::step])
    axes[1].set_xticklabels(labels[::step])
    handles0, labels0 = axes[0].get_legend_handles_labels()
    handles1, labels1 = axes[1].get_legend_handles_labels()
    fig.legend(handles0 + handles1, labels0 + labels1, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01), fontsize=7)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.9], h_pad=2.0)
    save(fig, out_dir, "Fig_5_case_performance_curves_no_fifo", no_write)
    plt.close(fig)


def violin_box(ax, values: Sequence[Sequence[float]], labels: Sequence[str]) -> None:
    parts = ax.violinplot(values, showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.3)
    ax.boxplot(values, labels=labels, widths=0.22, showfliers=False)


def plot_distribution(data, out_dir, no_write, methods, stem, zoom=False):
    rows = data["case"]
    values, labels = [], []
    for method in methods:
        group = [fnum(row.get("ARPD_no_fifo")) for row in rows if row.get("method") == method]
        if group:
            values.append(group)
            labels.append(method)
    if not values:
        warn(stem, "no no-FIFO ARPD data")
        return
    plt = mpl()
    fig, ax = plt.subplots(figsize=(max(5.8, len(labels) * 1.0), 3.4))
    violin_box(ax, values, labels)
    if zoom:
        flat = sorted(value for group in values for value in group)
        upper = flat[min(len(flat) - 1, int(0.95 * (len(flat) - 1)))]
        margin = max(0.3, (upper - min(flat)) * 0.15)
        ax.set_ylim(min(-0.2, min(flat) - margin), upper + margin)
    ax.set_ylabel("ARPD (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    save(fig, out_dir, stem, no_write)
    plt.close(fig)


def plot_fig6(data, out_dir, no_write, top_methods_zoom):
    plot_distribution(data, out_dir, no_write, DISTRIBUTION_METHOD_ORDER, "Fig_6a_distribution_all_methods_no_fifo")
    if top_methods_zoom:
        plot_distribution(data, out_dir, no_write, top_methods_zoom, "Fig_6b_distribution_top_methods_zoom_no_fifo", zoom=True)


def plot_fig8(data, out_dir, no_write):
    rows = data["scale"]
    if not rows:
        warn("Fig_8", "missing v3 scale summary")
        return
    import numpy as np

    plt = mpl()
    sizes = [*SIZES, "all"]
    x = np.arange(len(sizes))
    width = 0.15
    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    for idx, method in enumerate(METHODS):
        y = []
        for size in sizes:
            hit = next((row for row in rows if row.get("size") == size and row.get("method") == method), None)
            y.append(fnum(hit.get("ARPD_mean")) if hit else float("nan"))
        ax.bar(x + (idx - 2) * width, y, width=width, label=method, color=COLORS[method])
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("ARPD mean (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=7)
    fig.tight_layout()
    save(fig, out_dir, "Fig_8_scale_generalization_no_fifo", no_write)
    plt.close(fig)


def plot_a1(data, out_dir, no_write, compare_baselines):
    rows = data["summary"]
    if not rows:
        warn("Fig_A1", "missing v3 summary")
        return
    import numpy as np

    plt = mpl()
    fig, axes = plt.subplots(1, len(compare_baselines), figsize=(3.2 * len(compare_baselines), 3.1), sharey=True)
    axes = np.atleast_1d(axes).ravel().tolist()
    any_missing = False
    image = None
    for ax, baseline in zip(axes, compare_baselines):
        matrix = np.full((3, 3), np.nan)
        missing = []
        for i, arrival in enumerate(ARRIVALS):
            for j, carryover in enumerate(CARRYOVERS):
                hgcr = next((row for row in rows if row.get("size") == "small" and row.get("arrival_intensity") == arrival and row.get("carryover_ratio") == carryover and round(fnum(row.get("reward_beta")), 6) == 5.0 and row.get("method") == "HGCR-PPO"), None)
                base = next((row for row in rows if row.get("size") == "small" and row.get("arrival_intensity") == arrival and row.get("carryover_ratio") == carryover and round(fnum(row.get("reward_beta")), 6) == 5.0 and row.get("method") == baseline), None)
                if not hgcr or not base:
                    missing.append((arrival, carryover))
                    continue
                matrix[i, j] = (fnum(base["Cmax_mean"]) - fnum(hgcr["Cmax_mean"])) / max(fnum(base["Cmax_mean"]), 1e-8) * 100.0
        if missing:
            any_missing = True
            print(f"Missing Fig_A1 scenarios vs {baseline}: {missing}")
        vmax = max(1.0, float(np.nanmax(np.abs(matrix)))) if not np.isnan(matrix).all() else 1.0
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"HGCR-PPO vs {baseline}")
        ax.set_xticks(range(3)); ax.set_xticklabels(CARRYOVERS)
        ax.set_yticks(range(3)); ax.set_yticklabels(ARRIVALS)
        ax.set_xlabel("Carryover")
        for i in range(3):
            for j in range(3):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("Arrival")
    if image is not None:
        fig.colorbar(image, ax=axes, label="Improvement (%)")
    stem = "Fig_A1_dynamic_heatmap_3x3_no_fifo_partial" if any_missing else "Fig_A1_dynamic_heatmap_3x3_no_fifo"
    save(fig, out_dir, stem, no_write)
    plt.close(fig)


def plot_a2(data, out_dir, no_write, compare_baselines):
    summary = [row for row in data["summary"] if row.get("size") == "small" and row.get("arrival_intensity") == "medium" and row.get("carryover_ratio") == "medium"]
    hgcr = [row for row in summary if row.get("method") == "HGCR-PPO"]
    available = {round(fnum(row.get("reward_beta")), 6) for row in hgcr}
    missing = [beta for beta in BETAS if round(beta, 6) not in available]
    if missing:
        warn("Fig_A2", f"missing beta {missing}")
        return
    action_rows = [row for row in data["action_stage"] if row.get("size") == "small" and row.get("arrival_intensity") == "medium" and row.get("carryover_ratio") == "medium"]
    has_actions = bool(action_rows)
    import numpy as np

    plt = mpl()
    fig, axes = plt.subplots(1, 3 if has_actions else 2, figsize=(12.4 if has_actions else 7.6, 3.4))
    axes = np.ravel(axes).tolist()
    x = np.arange(len(BETAS))
    cmax = [fnum(next(row for row in hgcr if round(fnum(row.get("reward_beta")), 6) == round(beta, 6))["Cmax_mean"]) for beta in BETAS]
    axes[0].plot(x, cmax, marker="o", color=COLORS["HGCR-PPO"])
    axes[0].set_title("(a) beta -> final Cmax")
    axes[0].set_ylabel("Cmax mean")
    for baseline in compare_baselines:
        values = []
        for beta, hgcr_value in zip(BETAS, cmax):
            base = next((row for row in summary if row.get("method") == baseline and round(fnum(row.get("reward_beta")), 6) == round(beta, 6)), None)
            values.append((fnum(base["Cmax_mean"]) - hgcr_value) / max(fnum(base["Cmax_mean"]), 1e-8) * 100.0 if base else float("nan"))
        axes[1].plot(x, values, marker=".", label=f"vs {baseline}", color=COLORS[baseline])
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_title("(b) improvement over baselines")
    axes[1].set_ylabel("Improvement (%)")
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    if has_actions:
        bottom = np.zeros(len(BETAS))
        for action in ACTIONS:
            values = []
            for beta in BETAS:
                selected = [row for row in action_rows if round(fnum(row.get("reward_beta")), 6) == round(beta, 6) and row.get("action_name") == action]
                if selected:
                    last_end = max(fnum(row.get("stage_end_episode")) for row in selected)
                    selected = [row for row in selected if fnum(row.get("stage_end_episode")) == last_end]
                values.append(mean([fnum(row.get("action_ratio")) for row in selected]) if selected else 0.0)
            axes[2].bar(x, values, bottom=bottom, color=COLORS.get(action), label=ACTION_LABELS[action])
            bottom += np.asarray(values)
        axes[2].set_title("(c) final action ratio")
        axes[2].set_ylabel("Action ratio")
        handles, labels = axes[2].get_legend_handles_labels()
        axes[2].legend(handles, labels, frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([str(beta) for beta in BETAS], rotation=25)
        ax.set_xlabel("reward_beta")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 0.92 if has_actions else 1, 1], w_pad=2.0)
    save(fig, out_dir, "Fig_A2_beta_final_summary_no_fifo", no_write)
    plt.close(fig)


def plot_a3(data, out_dir, no_write, tie_threshold):
    rows = data["case"]
    if not rows:
        warn("Fig_A3", "missing v3 case detail")
        return
    results = []
    for baseline in BASELINES:
        counts = Counter({"Win": 0, "Tie": 0, "Loss": 0})
        labels = {row["case_label"] for row in rows if row.get("method") == "HGCR-PPO"}
        for label in labels:
            hgcr = next((row for row in rows if row.get("case_label") == label and row.get("method") == "HGCR-PPO"), None)
            base = next((row for row in rows if row.get("case_label") == label and row.get("method") == baseline), None)
            if not hgcr or not base:
                continue
            relative = (fnum(hgcr["Cmax"]) - fnum(base["Cmax"])) / max(fnum(base["Cmax"]), 1e-8)
            counts["Tie" if abs(relative) <= tie_threshold else ("Win" if relative < 0 else "Loss")] += 1
        results.append((baseline, counts))
    plt = mpl()
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    left = [0] * len(results)
    for label, color in [("Win", COLORS["HGCR-PPO"]), ("Tie", COLORS["auxiliary"]), ("Loss", COLORS["MLP-Ranker"])]:
        values = [counts[label] for _, counts in results]
        ax.barh([baseline for baseline, _ in results], values, left=left, label=label, color=color)
        left = [a + b for a, b in zip(left, values)]
    ax.set_xlabel("Number of cases")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, out_dir, "Fig_A3_win_tie_loss_no_fifo", no_write)
    plt.close(fig)


def plot_a4(data, out_dir, no_write):
    scale = [row for row in data["scale"] if row.get("size") == "all" and row.get("method") in METHODS]
    sig = data["significance"]
    if not scale:
        warn("Fig_A4", "missing all-size v3 scale summary")
        return
    plt = mpl()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    ordered = [next((row for row in scale if row.get("method") == method), None) for method in MAIN_METHOD_ORDER]
    methods = [method for method, row in zip(MAIN_METHOD_ORDER, ordered) if row]
    rows = [row for row in ordered if row]
    axes[0].bar(methods, [fnum(row["ARPD_mean"]) for row in rows], color=[COLORS[method] for method in methods])
    axes[1].bar(methods, [fnum(row["rank_mean"]) for row in rows], color=[COLORS[method] for method in methods])
    axes[0].set_title("(a) ARPD mean")
    axes[1].set_title("(b) Mean rank")
    axes[0].set_ylabel("ARPD (%)")
    axes[1].set_ylabel("Rank")
    for ax in axes[:2]:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(w_pad=2.0)
    save(fig, out_dir, "Fig_A4_arpd_rank_no_fifo", no_write)
    source_rows = [{key: row.get(key, "") for key in ["size", "method", "Cmax_mean", "Cmax_std", "ARPD_mean", "ARPD_std", "rank_mean", "rank_std", "n_instances"]} for row in rows]
    sig_rows = [{key: row.get(key, "") for key in ["comparison", "baseline_method", "test_name", "n_pairs", "mean_diff", "median_diff", "p_value", "significant", "effect_direction"]} for row in sig]
    write_csv(out_dir / "Fig_A4_arpd_rank_source_no_fifo.csv", source_rows, ["size", "method", "Cmax_mean", "Cmax_std", "ARPD_mean", "ARPD_std", "rank_mean", "rank_std", "n_instances"], no_write)
    write_csv(out_dir / "Fig_A4_significance_table_no_fifo.csv", sig_rows, ["comparison", "baseline_method", "test_name", "n_pairs", "mean_diff", "median_diff", "p_value", "significant", "effect_direction"], no_write)
    plt.close(fig)


def run(args):
    compare_baselines = validated_methods(args.main_compare_baselines)
    top_methods_zoom = validated_methods(args.top_methods_zoom, include_hgcr=True)
    data = discover(args)
    out_dir = Path(args.output_dir) / token()
    print(f"Planned figure dir: {out_dir}")
    print(f"Input files: {data['files']}")
    if data["missing_v3"]:
        message = f"Missing required v3 no-FIFO files: {data['missing_v3']}; no v2 fallback is allowed."
        print(f"Error: {message}")
        if not args.dry_run:
            raise FileNotFoundError(message)
        return out_dir
    no_write = args.no_write or args.dry_run
    if not no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
    plot_fig2(data, out_dir, no_write, args)
    plot_fig3(data, out_dir, no_write, args.smoothing_window, args.show_raw_curves)
    plot_fig4(data, out_dir, no_write)
    plot_fig5(data, out_dir, no_write, args.max_case_labels, compare_baselines)
    plot_fig6(data, out_dir, no_write, top_methods_zoom if args.generate_zoom_distribution else [])
    plot_fig8(data, out_dir, no_write)
    plot_a1(data, out_dir, no_write, compare_baselines)
    plot_a2(data, out_dir, no_write, compare_baselines)
    plot_a3(data, out_dir, no_write, args.tie_threshold)
    plot_a4(data, out_dir, no_write)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_dir", default=str(PAPER_DIR))
    parser.add_argument("--runs_dir", default=str(RUNS_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--smoothing_window", type=int, default=3)
    parser.add_argument("--show_raw_curves", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split_convergence_by_size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--convergence_sizes", nargs="+", default=["small"])
    parser.add_argument("--require_complete_convergence_runs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_required_episode", type=int, default=5000)
    parser.add_argument("--allow_incomplete_raw_curves", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude_incomplete_from_mean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--main_compare_baselines", nargs="+", default=BASELINES)
    parser.add_argument("--top_methods_zoom", nargs="+", default=["HGCR-PPO", "MinLoad", "GreedyECT", "Lookahead"])
    parser.add_argument("--generate_zoom_distribution", action="store_true")
    parser.add_argument("--max_case_labels", type=int, default=60)
    parser.add_argument("--tie_threshold", type=float, default=0.001)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
