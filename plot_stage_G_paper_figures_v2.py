"""Stage G-Plus v2 paper figures.

This script intentionally avoids the old summary-only plotting logic. Curves
use eval_history/action_stage_summary with real episode axes; case figures use
per-instance detail tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PAPER_DIR = Path("data/results/stage_G/paper_results")
RUNS_DIR = Path("data/results/stage_G/hgcr_dynamic_ppo/runs")
OUTPUT_DIR = Path("data/results/stage_G/paper_figures")
GANTT_DIR = Path("data/results/stage_G/gantt_cases")
ARRIVALS = ["low", "medium", "high"]
CARRYOVERS = ["low", "medium", "high"]
BETAS = [0.01, 0.1, 1.0, 2.0, 5.0]
METHODS = ["FIFO", "GreedyECT", "Lookahead", "MLP-Ranker", "HGCR-PPO"]
ACTIONS = ["FIFO", "GreedyECT", "Lookahead", "MLP-Ranker"]


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


def fnum(value, default=0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(value) or math.isinf(value) else value


def mpl():
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
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
    window = max(1, window)
    out, buf, total = [], [], 0.0
    for v in vals:
        buf.append(v)
        total += v
        if len(buf) > window:
            total -= buf.pop(0)
        out.append(total / len(buf))
    return out


def discover(args) -> Dict[str, object]:
    paper = Path(args.paper_dir)
    runs = Path(args.runs_dir)
    files = {
        "summary": latest(paper, "stage_G_method_comparison_summary_v2__*.csv"),
        "arpd": latest(paper, "stage_G_arpd_summary_v2__*.csv"),
        "sig": latest(paper, "stage_G_significance_tests_v2__*.csv"),
        "case": latest(paper, "stage_G_case_curve_detail_v2__*.csv"),
        "detail": latest(paper, "stage_G_method_comparison_detail__*.csv"),
        "trace": latest(paper, "schedule_trace__*.csv"),
    }
    data = {k: read_csv(v) for k, v in files.items()}
    data["eval_history"] = [row for path in runs.glob("*/eval_history*.csv") for row in read_csv(path)]
    data["action_stage"] = [row for path in runs.glob("*/action_stage_summary*.csv") for row in read_csv(path)]
    data["files"] = files
    return data


def warn_missing(name: str, details: str) -> None:
    print(f"Warning: skip {name}: {details}")


def filter_hist(rows: List[dict], *, arrival="medium", carryover="medium", beta=None, seed=None) -> List[dict]:
    out = []
    for r in rows:
        if r.get("arrival_intensity") != arrival or r.get("carryover_ratio") != carryover:
            continue
        if beta is not None and round(fnum(r.get("reward_beta")), 6) != round(beta, 6):
            continue
        if seed is not None and str(r.get("seed")) != str(seed):
            continue
        out.append(r)
    return sorted(out, key=lambda r: fnum(r.get("episode")))


def plot_fig2(data, out_dir, no_write, smooth):
    rows = filter_hist(data["eval_history"], beta=5.0)
    seeds = sorted({str(r.get("seed")) for r in rows})
    missing = [s for s in ["0", "1", "2"] if s not in seeds]
    if not rows or missing:
        warn_missing("Fig_2_training_convergence", f"missing eval_history for seeds {missing}")
        return
    import numpy as np

    plt = mpl()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.1), sharex=True)
    metrics = [("eval_Cmax_mean", "Eval Cmax"), ("eval_reward_mean", "Eval reward"), ("best_so_far_Cmax", "Best-so-far Cmax")]
    for ax, (metric, ylabel) in zip(axes, metrics):
        curves = []
        x_ref = None
        for seed in ["0", "1", "2"]:
            vals = filter_hist(rows, beta=5.0, seed=seed)
            x = [fnum(r["episode"]) for r in vals]
            y = [fnum(r[metric]) for r in vals]
            if x_ref is None:
                x_ref = x
            curves.append(y)
            ax.plot(x, y, alpha=0.35, linewidth=1.0, label=f"seed{seed}")
        min_len = min(len(c) for c in curves)
        avg = np.mean([c[:min_len] for c in curves], axis=0)
        ax.plot(x_ref[:min_len], moving_average(avg.tolist(), smooth), color="#111111", linewidth=2.0, label="mean")
        if metric == "eval_Cmax_mean":
            fifo = [fnum(r.get("baseline_FIFO_Cmax")) for r in rows if r.get("baseline_FIFO_Cmax")]
            mlp = [fnum(r.get("baseline_MLPRanker_Cmax")) for r in rows if r.get("baseline_MLPRanker_Cmax")]
            if fifo:
                ax.axhline(sum(fifo) / len(fifo), linestyle="--", color="#666666", label="FIFO")
            if mlp:
                ax.axhline(sum(mlp) / len(mlp), linestyle=":", color="#AA3333", label="MLP-Ranker")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, out_dir, "Fig_2_training_convergence", no_write)
    plt.close(fig)


def plot_fig3(data, out_dir, no_write, smooth):
    rows = filter_hist(data["eval_history"], seed=0)
    available = sorted({round(fnum(r.get("reward_beta")), 6) for r in rows})
    missing = [b for b in BETAS if round(b, 6) not in available]
    if missing:
        warn_missing("Fig_3_beta_sensitivity_training_curves", f"missing beta runs {missing}")
        return
    plt = mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharex=True)
    for beta in BETAS:
        vals = filter_hist(rows, beta=beta, seed=0)
        x = [fnum(r["episode"]) for r in vals]
        cmax = moving_average([fnum(r["eval_Cmax_mean"]) for r in vals], smooth)
        reward = moving_average([fnum(r["eval_reward_mean"]) for r in vals], smooth)
        axes[0].plot(x, cmax, label=f"beta={beta:g}")
        axes[1].plot(x, reward, label=f"beta={beta:g}")
    axes[0].set_ylabel("Eval Cmax")
    axes[1].set_ylabel("Eval reward")
    for ax in axes:
        ax.set_xlabel("Episode")
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, out_dir, "Fig_3_beta_sensitivity_training_curves", no_write)
    plt.close(fig)


def plot_fig4(data, out_dir, no_write):
    rows = [r for r in data["action_stage"] if r.get("arrival_intensity") == "medium" and r.get("carryover_ratio") == "medium" and round(fnum(r.get("reward_beta")), 6) == 5.0]
    seeds = sorted({str(r.get("seed")) for r in rows})
    missing = [s for s in ["0", "1", "2"] if s not in seeds]
    if not rows or missing:
        warn_missing("Fig_4_action_ratio_evolution", f"missing action_stage_summary for seeds {missing}")
        return
    import numpy as np

    plt = mpl()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True)
    for ax, seed in zip(axes, ["0", "1", "2"]):
        vals = [r for r in rows if str(r.get("seed")) == seed]
        stages = sorted({f"{r['stage_start_episode']}-{r['stage_end_episode']}" for r in vals}, key=lambda s: int(s.split("-")[0]))
        x = np.arange(len(stages))
        bottom = np.zeros(len(stages))
        for action in ACTIONS:
            y = []
            for st in stages:
                start, end = st.split("-")
                hit = next((r for r in vals if r.get("action_name") == action and r.get("stage_start_episode") == start and r.get("stage_end_episode") == end), None)
                y.append(fnum(hit.get("action_ratio")) if hit else 0.0)
            ax.bar(x, y, bottom=bottom, label=action)
            bottom += np.asarray(y)
        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=35, ha="right")
        ax.set_title(f"seed{seed}")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Action percentage")
    axes[1].legend(frameon=False, ncol=4, bbox_to_anchor=(0.5, 1.25), loc="upper center")
    fig.tight_layout()
    save(fig, out_dir, "Fig_4_action_ratio_evolution", no_write)
    plt.close(fig)


def plot_fig5(data, out_dir, no_write):
    rows = data["case"]
    if not rows:
        warn_missing("Fig_5_case_performance_curves", "missing stage_G_case_curve_detail_v2")
        return
    cases = sorted({r["case_id"] for r in rows})[:50]
    if len(cases) < 30:
        print(f"Warning: Fig_5 has only {len(cases)} cases; target is at least 30.")
    plt = mpl()
    fig, axes = plt.subplots(2, 1, figsize=(max(8.0, 0.16 * len(cases)), 5.2), sharex=True)
    x = list(range(len(cases)))
    for method in METHODS:
        y = []
        for cid in cases:
            hit = next((r for r in rows if r["case_id"] == cid and r["method"] == method), None)
            y.append(fnum(hit["Cmax"]) if hit else float("nan"))
        axes[0].plot(x, y, marker=".", linewidth=1.0, label=method)
    hgcr_imp = []
    for cid in cases:
        hit = next((r for r in rows if r["case_id"] == cid and r["method"] == "HGCR-PPO"), None)
        hgcr_imp.append(fnum(hit["improvement_vs_FIFO"]) if hit else 0.0)
    axes[1].bar(x, hgcr_imp, color="#2F6BFF")
    axes[0].set_ylabel("Cmax")
    axes[1].set_ylabel("Improvement vs FIFO (%)")
    axes[1].set_xticks(x[:: max(1, len(x)//20)])
    axes[1].set_xticklabels([cases[i] for i in x[:: max(1, len(x)//20)]], rotation=45, ha="right")
    axes[0].legend(frameon=False, ncol=5, fontsize=7)
    fig.tight_layout()
    save(fig, out_dir, "Fig_5_case_performance_curves", no_write)
    plt.close(fig)


def plot_fig6(data, out_dir, no_write):
    rows = data["case"]
    if not rows:
        warn_missing("Fig_6_distribution_by_scenario", "missing case curve detail")
        return
    plt = mpl()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, arrival in zip(axes, ARRIVALS):
        vals, labels = [], []
        for method in METHODS:
            y = []
            for r in rows:
                if r["arrival_intensity"] == arrival and r["method"] == method:
                    best = fnum(r["Cmax_best_among_selected_methods"])
                    y.append((fnum(r["Cmax"]) - best) / max(best, 1e-8) * 100.0)
            if y:
                vals.append(y)
                labels.append(method)
        if vals:
            ax.boxplot(vals, labels=labels, showfliers=False)
        ax.set_title(f"arrival={arrival}")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("ARPD (%)")
    fig.tight_layout()
    save(fig, out_dir, "Fig_6_distribution_by_scenario", no_write)
    plt.close(fig)


def plot_a1(data, out_dir, no_write):
    summary = data["summary"]
    combos = {(r["arrival_intensity"], r["carryover_ratio"]) for r in summary if r["method"] == "HGCR-PPO"}
    missing = [(a, c) for a in ARRIVALS for c in CARRYOVERS if (a, c) not in combos]
    if missing:
        warn_missing("Fig_A1_dynamic_heatmap_3x3", f"missing scenario combos {missing}")
        return
    import numpy as np

    plt = mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, baseline, title in [(axes[0], "FIFO", "HGCR vs FIFO"), (axes[1], "MLP-Ranker", "HGCR vs MLP-Ranker")]:
        mat = np.full((3, 3), np.nan)
        for i, a in enumerate(ARRIVALS):
            for j, c in enumerate(CARRYOVERS):
                hgcr = next(r for r in summary if r["arrival_intensity"] == a and r["carryover_ratio"] == c and r["method"] == "HGCR-PPO")
                base = next(r for r in summary if r["arrival_intensity"] == a and r["carryover_ratio"] == c and r["method"] == baseline)
                mat[i, j] = (fnum(base["Cmax_mean"]) - fnum(hgcr["Cmax_mean"])) / max(fnum(base["Cmax_mean"]), 1e-8) * 100.0
        vmax = max(1.0, float(np.nanmax(np.abs(mat))))
        im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(3)); ax.set_xticklabels(CARRYOVERS)
        ax.set_yticks(range(3)); ax.set_yticklabels(ARRIVALS)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]:.1f}%", ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("Arrival")
    for ax in axes:
        ax.set_xlabel("Carryover")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Improvement (%)")
    save(fig, out_dir, "Fig_A1_dynamic_heatmap_3x3", no_write)
    plt.close(fig)


def plot_a2(data, out_dir, no_write):
    rows = [r for r in data["summary"] if r.get("arrival_intensity") == "medium" and r.get("carryover_ratio") == "medium" and r.get("method") == "HGCR-PPO"]
    betas = sorted({round(fnum(r.get("reward_beta")), 6) for r in rows})
    missing = [b for b in BETAS if round(b, 6) not in betas]
    if missing:
        warn_missing("Fig_A2_beta_final_summary", f"missing beta {missing}")
        return
    import numpy as np

    plt = mpl()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0))
    ordered = []
    for beta in BETAS:
        ordered.append(next(r for r in rows if round(fnum(r.get("reward_beta")), 6) == round(beta, 6)))
    x = np.arange(len(BETAS))
    axes[0].plot(x, [fnum(r["Cmax_mean"]) for r in ordered], marker="o")
    axes[0].set_ylabel("HGCR Cmax")
    fifo_rows = [r for r in data["summary"] if r.get("arrival_intensity") == "medium" and r.get("carryover_ratio") == "medium" and r.get("method") == "FIFO"]
    fifo_map = {round(fnum(r.get("reward_beta")), 6): fnum(r["Cmax_mean"]) for r in fifo_rows}
    imp = [(fifo_map.get(round(beta, 6), fnum(ordered[i]["Cmax_mean"])) - fnum(ordered[i]["Cmax_mean"])) / max(fifo_map.get(round(beta, 6), 1.0), 1e-8) * 100.0 for i, beta in enumerate(BETAS)]
    axes[1].bar(x, imp)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Improvement vs FIFO (%)")
    axes[2].text(0.1, 0.5, "Final action-ratio panel uses\naction_stage_summary/action_history\nwhen available.", fontsize=9)
    axes[2].axis("off")
    for ax in axes[:2]:
        ax.set_xticks(x); ax.set_xticklabels([str(b) for b in BETAS], rotation=30)
        ax.set_xlabel("reward_beta")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, out_dir, "Fig_A2_beta_final_summary", no_write)
    plt.close(fig)


def plot_a3_a4(data, out_dir, no_write):
    if not data["sig"]:
        warn_missing("Fig_A3_win_tie_loss", "missing significance/win-tie-loss inputs")
    if not data["arpd"]:
        warn_missing("Fig_A4_arpd_rank_significance", "missing ARPD inputs")


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
    for fn in [plot_fig2, plot_fig3]:
        fn(data, out_dir, no_write, args.smoothing_window)
    for fn in [plot_fig4, plot_fig5, plot_fig6, plot_a1, plot_a2, plot_a3_a4]:
        fn(data, out_dir, no_write)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_dir", default=str(PAPER_DIR))
    parser.add_argument("--runs_dir", default=str(RUNS_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--smoothing_window", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
