"""Stage G-Plus v2 paper figures rebuilt around training, case, and scale views."""

from __future__ import annotations

import argparse
import csv
import math
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence


PAPER_DIR = Path("data/results/stage_G/paper_results")
RUNS_DIR = Path("data/results/stage_G/hgcr_dynamic_ppo/runs")
OUTPUT_DIR = Path("data/results/stage_G/paper_figures")
ARRIVALS = ["low", "medium", "high"]
CARRYOVERS = ["low", "medium", "high"]
SIZES = ["small", "medium", "large"]
BETAS = [0.01, 0.1, 1.0, 2.0, 5.0]
ACTIONS = ["FIFO", "GreedyECT", "Lookahead", "MLP-Ranker"]
CASE_METHODS = ["FIFO", "GreedyECT", "Lookahead", "MLP-Ranker", "HGCR-PPO"]
COLORS = {
    "FIFO": "#6F6F6F",
    "GreedyECT": "#1F77B4",
    "Lookahead": "#2CA02C",
    "MinLoad": "#9467BD",
    "MLP-Ranker": "#D62728",
    "HGCR-PPO": "#111111",
}


def token() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    return matches[-1] if matches else None


def read_csv(path: Path | None) -> List[dict]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str], no_write: bool) -> None:
    print(f"Plan CSV: {path}")
    if no_write:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value, default=0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(value) or math.isinf(value) else value


def mpl():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.titlesize": 10,
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


def moving_average(vals: Sequence[float], window: int) -> List[float]:
    window = max(1, int(window))
    out: List[float] = []
    for idx in range(len(vals)):
        start = max(0, idx - window + 1)
        out.append(mean(vals[start : idx + 1]))
    return out


def warn_missing(name: str, details: str) -> None:
    print(f"Warning: skip {name}: {details}")


def discover(args) -> Dict[str, object]:
    paper = Path(args.paper_dir)
    runs = Path(args.runs_dir)
    files = {
        "summary": latest(paper, "stage_G_method_comparison_summary_v2__*.csv"),
        "arpd": latest(paper, "stage_G_arpd_summary_v2__*.csv"),
        "sig": latest(paper, "stage_G_significance_tests_v2__*.csv"),
        "case": latest(paper, "stage_G_case_curve_detail_v2__*.csv"),
        "scale": latest(paper, "stage_G_scale_summary_v2__*.csv"),
        "case_mapping": latest(paper, "stage_G_case_mapping_v2__*.csv"),
        "detail": latest(paper, "stage_G_method_comparison_detail__*.csv"),
        "trace": latest(paper, "schedule_trace__*.csv"),
    }
    data = {key: read_csv(path) for key, path in files.items()}
    data["eval_history"] = [row for path in runs.glob("*/eval_history*.csv") for row in read_csv(path)]
    data["action_stage"] = [row for path in runs.glob("*/action_stage_summary*.csv") for row in read_csv(path)]
    data["files"] = files
    return data


def split_methods(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        return [item for item in value.split() if item]
    return list(value)


def filter_hist(rows: List[dict], *, size=None, arrival="medium", carryover="medium", beta=None, seed=None) -> List[dict]:
    out = []
    for row in rows:
        if size is not None and row.get("size", "small") != size:
            continue
        if row.get("arrival_intensity") != arrival or row.get("carryover_ratio") != carryover:
            continue
        if beta is not None and round(fnum(row.get("reward_beta")), 6) != round(beta, 6):
            continue
        if seed is not None and str(row.get("seed")) != str(seed):
            continue
        out.append(row)
    return sorted(out, key=lambda r: fnum(r.get("episode")))


def plot_training_metric(data, out_dir: Path, no_write: bool, smooth: int, show_raw: bool, metric: str, stem: str, ylabel: str) -> None:
    rows = data["eval_history"]
    available_sizes = [size for size in SIZES if filter_hist(rows, size=size, beta=5.0)]
    if not available_sizes:
        warn_missing(stem, "missing medium-medium beta=5.0 eval_history for small/medium/large")
        return
    plt = mpl()
    fig, axes = plt.subplots(1, len(available_sizes), figsize=(4.0 * len(available_sizes), 3.2), sharey=False)
    axes = axes if isinstance(axes, (list, tuple)) else [axes]
    try:
        import numpy as np

        axes = np.ravel(axes).tolist()
    except Exception:
        pass
    for ax, size in zip(axes, available_sizes):
        curves, x_refs = [], []
        missing = []
        for seed in ["0", "1", "2"]:
            vals = filter_hist(rows, size=size, beta=5.0, seed=seed)
            if not vals:
                missing.append(seed)
                continue
            x = [fnum(r["episode"]) for r in vals]
            y = [fnum(r[metric]) for r in vals]
            x_refs.append(x)
            curves.append(y)
            if show_raw:
                ax.plot(x, y, color=COLORS["HGCR-PPO"], alpha=0.22, linewidth=0.8, label=f"seed{seed} raw")
        if missing:
            print(f"Warning: {stem} size={size} missing seeds {missing}")
        if curves:
            min_len = min(len(c) for c in curves)
            avg = [mean(c[idx] for c in curves if len(c) > idx) for idx in range(min_len)]
            ax.plot(x_refs[0][:min_len], moving_average(avg, smooth), color=COLORS["HGCR-PPO"], linewidth=1.8, label="seed mean smoothed")
            if metric == "eval_Cmax_mean":
                baseline_rows = [r for r in filter_hist(rows, size=size, beta=5.0) if r.get("baseline_FIFO_Cmax")]
                fifo = [fnum(r.get("baseline_FIFO_Cmax")) for r in baseline_rows]
                mlp = [fnum(r.get("baseline_MLPRanker_Cmax")) for r in baseline_rows]
                if fifo:
                    ax.axhline(mean(fifo), color=COLORS["FIFO"], linestyle="--", linewidth=1.0, label="FIFO baseline")
                if mlp:
                    ax.axhline(mean(mlp), color=COLORS["MLP-Ranker"], linestyle=":", linewidth=1.1, label="MLP-Ranker baseline")
        ax.set_title(f"({chr(97 + SIZES.index(size))}) {size}")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, out_dir, stem, no_write)
    plt.close(fig)


def plot_fig2(data, out_dir, no_write, smooth, show_raw):
    plot_training_metric(data, out_dir, no_write, smooth, show_raw, "eval_Cmax_mean", "Fig_2_training_convergence", "Eval Cmax mean")
    plot_training_metric(data, out_dir, no_write, smooth, show_raw, "eval_reward_mean", "Fig_2b_training_reward_convergence", "Eval reward mean")


def plot_fig3(data, out_dir, no_write, smooth, show_raw):
    rows = filter_hist(data["eval_history"], size="small", seed=0)
    available = sorted({round(fnum(r.get("reward_beta")), 6) for r in rows})
    missing = [beta for beta in BETAS if round(beta, 6) not in available]
    if missing:
        print(f"Warning: Fig_3_beta_sensitivity_training_curves missing beta = {missing}")
    if not rows:
        warn_missing("Fig_3_beta_sensitivity_training_curves", "missing small medium-medium seed0 eval_history")
        return
    plt = mpl()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharex=True)
    palette = ["#7B3294", "#008837", "#A6611A", "#0571B0", "#CA0020"]
    for beta, color in zip(BETAS, palette):
        vals = filter_hist(rows, size="small", beta=beta, seed=0)
        if not vals:
            continue
        x = [fnum(r["episode"]) for r in vals]
        cmax = [fnum(r["eval_Cmax_mean"]) for r in vals]
        reward = [fnum(r["eval_reward_mean"]) for r in vals]
        if show_raw:
            axes[0].plot(x, cmax, color=color, alpha=0.22, linewidth=0.8)
            axes[1].plot(x, reward, color=color, alpha=0.22, linewidth=0.8)
        axes[0].plot(x, moving_average(cmax, smooth), color=color, linewidth=1.6, label=f"beta={beta:g}")
        axes[1].plot(x, moving_average(reward, smooth), color=color, linewidth=1.6, label=f"beta={beta:g}")
    axes[0].set_title("(a) Eval Cmax under different beta")
    axes[1].set_title("(b) Eval reward under different beta")
    axes[0].set_ylabel("Eval Cmax mean")
    axes[1].set_ylabel("Eval reward mean")
    for ax in axes:
        ax.set_xlabel("Episode")
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, out_dir, "Fig_3_beta_sensitivity_training_curves", no_write)
    plt.close(fig)


def plot_fig4(data, out_dir, no_write):
    rows = [
        r
        for r in data["action_stage"]
        if r.get("arrival_intensity") == "medium" and r.get("carryover_ratio") == "medium" and round(fnum(r.get("reward_beta")), 6) == 5.0
    ]
    available_sizes = [size for size in SIZES if any(r.get("size", "small") == size for r in rows)]
    panel_values = available_sizes if available_sizes else sorted({str(r.get("seed")) for r in rows})
    if not panel_values:
        warn_missing("Fig_4_action_ratio_evolution", "missing action_stage_summary")
        return
    plt = mpl()
    import numpy as np

    fig, axes = plt.subplots(1, len(panel_values), figsize=(4.1 * len(panel_values), 3.3), sharey=True)
    axes = np.ravel([axes]).tolist() if len(panel_values) == 1 else np.ravel(axes).tolist()
    for ax, panel in zip(axes, panel_values):
        vals = [r for r in rows if (r.get("size", "small") == panel if panel in SIZES else str(r.get("seed")) == panel)]
        stages = [(1, 1000), (1001, 2000), (2001, 3000), (3001, 4000), (4001, 5000)]
        labels = [f"{s - 1}-{e}" if s > 1 else f"0-{e}" for s, e in stages]
        x = np.arange(len(stages))
        bottom = np.zeros(len(stages))
        for action in ACTIONS:
            y = []
            for start, end in stages:
                hits = [r for r in vals if r.get("action_name") == action and int(float(r.get("stage_start_episode", 0))) >= start and int(float(r.get("stage_end_episode", 0))) <= end]
                y.append(mean([fnum(h.get("action_ratio")) for h in hits]) if hits else 0.0)
            ax.bar(x, y, bottom=bottom, color=COLORS.get(action), label=action)
            bottom += np.asarray(y)
        ax.set_title(panel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Action percentage")
    fig.suptitle("HGCR-PPO action selection evolution", y=1.03)
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    save(fig, out_dir, "Fig_4_action_ratio_evolution", no_write)
    plt.close(fig)


def ordered_cases(rows: List[dict], max_cases: int) -> List[str]:
    labels = sorted({r.get("case_label") or r["case_id"] for r in rows}, key=lambda v: int(str(v).lstrip("C") or 0) if str(v).startswith("C") else str(v))
    return labels[:max_cases]


def plot_fig5(data, out_dir, no_write, max_labels):
    rows = data["case"]
    if not rows:
        warn_missing("Fig_5_case_performance_curves", "missing stage_G_case_curve_detail_v2")
        return
    cases = ordered_cases(rows, max_labels)
    if len(cases) < 30:
        print(f"Warning: Fig_5 has only {len(cases)} cases; target is at least 30.")
    mapping = []
    seen = set()
    for r in rows:
        label = r.get("case_label") or r["case_id"]
        if label in cases and label not in seen:
            seen.add(label)
            mapping.append({k: r.get(k, "") for k in ["case_label", "case_id", "size", "arrival_intensity", "carryover_ratio", "seed", "instance_id"]})
    write_csv(out_dir / "Fig_5_case_mapping.csv", mapping, ["case_label", "case_id", "size", "arrival_intensity", "carryover_ratio", "seed", "instance_id"], no_write)
    plt = mpl()
    fig, axes = plt.subplots(2, 1, figsize=(max(9.0, 0.15 * len(cases)), 5.4), sharex=True)
    x = list(range(len(cases)))
    for method in CASE_METHODS:
        y = []
        for label in cases:
            hit = next((r for r in rows if (r.get("case_label") or r["case_id"]) == label and r["method"] == method), None)
            y.append(fnum(hit["Cmax"]) if hit else float("nan"))
        axes[0].plot(x, y, marker=".", linewidth=1.0, color=COLORS.get(method), label=method)
    for baseline, field, color, lw, alpha in [
        ("MLP-Ranker", "improvement_vs_MLPRanker", COLORS["MLP-Ranker"], 1.5, 0.95),
        ("GreedyECT", "improvement_vs_GreedyECT", COLORS["GreedyECT"], 1.5, 0.95),
        ("Lookahead", "improvement_vs_Lookahead", COLORS["Lookahead"], 1.5, 0.95),
        ("FIFO", "improvement_vs_FIFO", COLORS["FIFO"], 0.9, 0.55),
    ]:
        y = []
        for label in cases:
            hit = next((r for r in rows if (r.get("case_label") or r["case_id"]) == label and r["method"] == "HGCR-PPO"), None)
            y.append(fnum(hit.get(field)) if hit else float("nan"))
        axes[1].plot(x, y, linewidth=lw, alpha=alpha, color=color, label=f"HGCR-PPO vs {baseline}")
    axes[0].set_title("(a) Cmax across representative cases")
    axes[1].set_title("(b) Relative improvement over selected baselines")
    axes[0].set_ylabel("Cmax")
    axes[1].set_ylabel("Improvement (%)")
    step = max(1, len(cases) // 12)
    ticks = x[::step]
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([cases[i] for i in ticks], rotation=0)
    axes[0].legend(frameon=False, ncol=5, fontsize=7)
    axes[1].legend(frameon=False, ncol=2, fontsize=7)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, out_dir, "Fig_5_case_performance_curves", no_write)
    plt.close(fig)


def violin_or_box(ax, vals, labels):
    try:
        parts = ax.violinplot(vals, showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_alpha(0.35)
        ax.boxplot(vals, labels=labels, widths=0.22, showfliers=False)
    except Exception:
        ax.boxplot(vals, labels=labels, showfliers=False)


def plot_distribution(data, out_dir, no_write, methods, stem, zoom=False):
    rows = data["case"]
    if not rows:
        warn_missing(stem, "missing case curve detail")
        return
    vals, labels = [], []
    for method in methods:
        y = []
        for r in rows:
            if r["method"] == method:
                best = fnum(r["Cmax_best_among_selected_methods"])
                y.append((fnum(r["Cmax"]) - best) / max(best, 1e-8) * 100.0)
        if y:
            vals.append(y)
            labels.append(method)
    if not vals:
        warn_missing(stem, f"missing methods {methods}")
        return
    plt = mpl()
    fig, ax = plt.subplots(figsize=(max(5.8, 0.9 * len(labels)), 3.4))
    violin_or_box(ax, vals, labels)
    if zoom:
        flat = sorted(v for group in vals for v in group)
        upper = flat[min(len(flat) - 1, max(0, int(0.95 * (len(flat) - 1))))]
        lower = min(flat)
        margin = max(0.5, (upper - lower) * 0.15)
        ax.set_ylim(min(0.0, lower - margin), upper + margin)
    ax.set_ylabel("ARPD (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save(fig, out_dir, stem, no_write)
    plt.close(fig)


def plot_fig6(data, out_dir, no_write, top_methods):
    plot_distribution(data, out_dir, no_write, ["FIFO", "GreedyECT", "Lookahead", "MinLoad", "MLP-Ranker", "HGCR-PPO"], "Fig_6a_distribution_all_methods")
    plot_distribution(data, out_dir, no_write, top_methods, "Fig_6b_distribution_top_methods_zoom", zoom=True)


def plot_a1(data, out_dir, no_write):
    summary = data["summary"]
    if not summary:
        warn_missing("Fig_A1_dynamic_heatmap_3x3", "missing summary")
        return
    baselines = ["MLP-Ranker", "GreedyECT", "Lookahead", "FIFO"]
    import numpy as np

    plt = mpl()
    fig, axes = plt.subplots(1, len(baselines), figsize=(3.2 * len(baselines), 3.1), sharey=True)
    axes = np.ravel(axes).tolist()
    im = None
    for ax, baseline in zip(axes, baselines):
        mat = np.full((3, 3), np.nan)
        missing = []
        for i, arrival in enumerate(ARRIVALS):
            for j, carryover in enumerate(CARRYOVERS):
                hgcr = next((r for r in summary if r.get("size", "small") == "small" and r["arrival_intensity"] == arrival and r["carryover_ratio"] == carryover and r["method"] == "HGCR-PPO"), None)
                base = next((r for r in summary if r.get("size", "small") == "small" and r["arrival_intensity"] == arrival and r["carryover_ratio"] == carryover and r["method"] == baseline), None)
                if hgcr is None or base is None:
                    missing.append((arrival, carryover))
                    continue
                mat[i, j] = (fnum(base["Cmax_mean"]) - fnum(hgcr["Cmax_mean"])) / max(fnum(base["Cmax_mean"]), 1e-8) * 100.0
        if missing:
            print(f"Warning: Fig_A1 {baseline} missing scenario combos {sorted(set(missing))}")
            ax.axis("off")
            ax.set_title(f"vs {baseline}\nmissing data")
            continue
        vmax = max(1.0, float(np.nanmax(np.abs(mat))))
        im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"HGCR-PPO vs {baseline}")
        ax.set_xticks(range(3)); ax.set_xticklabels(CARRYOVERS)
        ax.set_yticks(range(3)); ax.set_yticklabels(ARRIVALS)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]:.1f}%", ha="center", va="center", fontsize=8)
        ax.set_xlabel("Carryover")
    axes[0].set_ylabel("Arrival")
    if im is not None:
        fig.colorbar(im, ax=[ax for ax in axes if ax.has_data()], label="Improvement (%)")
    else:
        warn_missing("Fig_A1_dynamic_heatmap_3x3", "all 3x3 heatmap inputs are missing")
    save(fig, out_dir, "Fig_A1_dynamic_heatmap_3x3", no_write)
    plt.close(fig)


def plot_a2(data, out_dir, no_write):
    summary = [r for r in data["summary"] if r.get("size", "small") == "small" and r.get("arrival_intensity") == "medium" and r.get("carryover_ratio") == "medium"]
    action_rows = [r for r in data["action_stage"] if r.get("size", "small") == "small" and r.get("arrival_intensity") == "medium" and r.get("carryover_ratio") == "medium"]
    hgcr_rows = [r for r in summary if r.get("method") == "HGCR-PPO"]
    available = sorted({round(fnum(r.get("reward_beta")), 6) for r in hgcr_rows})
    missing = [b for b in BETAS if round(b, 6) not in available]
    if missing:
        warn_missing("Fig_A2_beta_final_summary", f"missing beta {missing}")
        return
    import numpy as np

    has_action = bool(action_rows)
    plt = mpl()
    fig, axes = plt.subplots(1, 3 if has_action else 2, figsize=(10.8 if has_action else 7.2, 3.1))
    axes = np.ravel(axes).tolist()
    x = np.arange(len(BETAS))
    cmax = []
    for beta in BETAS:
        hit = next(r for r in hgcr_rows if round(fnum(r.get("reward_beta")), 6) == round(beta, 6))
        cmax.append(fnum(hit["Cmax_mean"]))
    axes[0].plot(x, cmax, marker="o", color=COLORS["HGCR-PPO"])
    axes[0].set_title("(a) beta -> final Cmax")
    axes[0].set_ylabel("Cmax mean")
    for baseline in ["MLP-Ranker", "GreedyECT", "Lookahead"]:
        vals = []
        for beta, hgcr_cmax in zip(BETAS, cmax):
            base = next((r for r in summary if r.get("method") == baseline and round(fnum(r.get("reward_beta")), 6) == round(beta, 6)), None)
            vals.append((fnum(base["Cmax_mean"]) - hgcr_cmax) / max(fnum(base["Cmax_mean"]), 1e-8) * 100.0 if base else float("nan"))
        axes[1].plot(x, vals, marker=".", label=f"vs {baseline}", color=COLORS.get(baseline))
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_title("(b) improvement over baselines")
    axes[1].set_ylabel("Improvement (%)")
    axes[1].legend(frameon=False, fontsize=7)
    if has_action:
        bottom = np.zeros(len(BETAS))
        for action in ACTIONS:
            vals = []
            for beta in BETAS:
                hits = [r for r in action_rows if round(fnum(r.get("reward_beta")), 6) == round(beta, 6) and r.get("action_name") == action]
                vals.append(mean([fnum(h.get("action_ratio")) for h in hits]) if hits else 0.0)
            axes[2].bar(x, vals, bottom=bottom, color=COLORS.get(action), label=action)
            bottom += np.asarray(vals)
        axes[2].set_title("(c) final action ratio")
        axes[2].set_ylabel("Action ratio")
        axes[2].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([str(b) for b in BETAS], rotation=25)
        ax.set_xlabel("reward_beta")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, out_dir, "Fig_A2_beta_final_summary", no_write)
    plt.close(fig)


def plot_fig8(data, out_dir, no_write):
    rows = data["scale"]
    if not rows:
        warn_missing("Fig_8_scale_generalization", "missing stage_G_scale_summary_v2")
        return
    methods = ["FIFO", "GreedyECT", "Lookahead", "MLP-Ranker", "HGCR-PPO"]
    sizes = ["small", "medium", "large", "all"]
    import numpy as np

    plt = mpl()
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = np.arange(len(sizes))
    width = 0.14
    for idx, method in enumerate(methods):
        y = []
        for size in sizes:
            hit = next((r for r in rows if r.get("size") == size and r.get("method") == method), None)
            y.append(fnum(hit.get("ARPD_mean")) if hit else float("nan"))
        ax.bar(x + (idx - 2) * width, y, width=width, label=method, color=COLORS.get(method))
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("ARPD mean (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=7)
    fig.tight_layout()
    save(fig, out_dir, "Fig_8_scale_generalization", no_write)
    plt.close(fig)


def run(args):
    out_dir = Path(args.output_dir) / token()
    data = discover(args)
    print(f"Planned figure dir: {out_dir}")
    print(f"Input files: {data['files']}")
    if args.dry_run:
        print("Dry run enabled: no figures will be written.")
    no_write = args.no_write or args.dry_run
    if not no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
    main_compare = split_methods(args.main_compare_baselines)
    top_methods = split_methods(args.top_methods_zoom)
    plot_fig2(data, out_dir, no_write, args.smoothing_window, args.show_raw_curves)
    plot_fig3(data, out_dir, no_write, args.smoothing_window, args.show_raw_curves)
    plot_fig4(data, out_dir, no_write)
    plot_fig5(data, out_dir, no_write, args.max_case_labels)
    plot_fig6(data, out_dir, no_write, top_methods)
    plot_a1(data, out_dir, no_write)
    plot_a2(data, out_dir, no_write)
    plot_fig8(data, out_dir, no_write)
    if main_compare:
        print(f"Main comparison baselines: {main_compare}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_dir", default=str(PAPER_DIR))
    parser.add_argument("--runs_dir", default=str(RUNS_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--smoothing_window", type=int, default=3)
    parser.add_argument("--show_raw_curves", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--main_compare_baselines", nargs="+", default=["MLP-Ranker", "GreedyECT", "Lookahead", "MinLoad"])
    parser.add_argument("--top_methods_zoom", nargs="+", default=["FIFO", "MinLoad", "HGCR-PPO"])
    parser.add_argument("--max_case_labels", type=int, default=60)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
