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
HGCR_RUNS_DIR = Path("data/results/stage_G/hgcr_dynamic_ppo/runs")
GANTT_DIR = Path("data/results/stage_G/gantt_cases")
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
        "arpd": latest(root, "stage_G_arpd_summary__*.csv"),
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
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    x = np.arange(len(ARRIVAL_ORDER))
    for method in METHOD_ORDER:
        y = []
        err = []
        for arrival in ARRIVAL_ORDER:
            vals = [row for row in rows if row.get("arrival_intensity") == arrival and row.get("method") == method]
            y.append(sum(fnum(row.get("Cmax_mean")) for row in vals) / len(vals) if vals else float("nan"))
            err.append(sum(fnum(row.get("Cmax_std")) for row in vals) / len(vals) if vals else 0.0)
        ax.errorbar(x, y, yerr=err, marker="o", linewidth=1.2, capsize=2.5, label=method)
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
    rows = data["heatmap"]
    if not rows:
        print("Warning: skip G4, missing heatmap.")
        return
    carryovers = sorted({row["carryover_ratio"] for row in rows})
    if len(carryovers) < 2:
        print("Warning: only one carryover level found; heatmap will be a thin matrix.")
    import numpy as np

    plt = mpl()
    matrix = np.full((len(ARRIVAL_ORDER), len(carryovers)), np.nan)
    for row in rows:
        if row["arrival_intensity"] in ARRIVAL_ORDER:
            r = ARRIVAL_ORDER.index(row["arrival_intensity"])
            c = carryovers.index(row["carryover_ratio"])
            matrix[r, c] = fnum(row["HGCR_improvement_over_MLP"]) * 100.0
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(carryovers)))
    ax.set_xticklabels(carryovers)
    ax.set_yticks(range(len(ARRIVAL_ORDER)))
    ax.set_yticklabels(ARRIVAL_ORDER)
    ax.set_xlabel("Carryover ratio")
    ax.set_ylabel("Arrival intensity")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Improvement over MLP-Ranker (%)")
    fig.tight_layout()
    save(fig, out_dir, "fig_G4_dynamic_scenario_heatmap", no_write)
    plt.close(fig)


def fig_g5(data, out_dir, no_write):
    rows = data["action"]
    if not rows:
        print("Warning: skip G5, missing action-performance summary.")
        return
    import numpy as np

    plt = mpl()
    grouped = {}
    for row in rows:
        key = (row["arrival_intensity"], row["carryover_ratio"])
        grouped.setdefault(key, []).append(row)
    rows = []
    for key, vals in grouped.items():
        base = dict(vals[0])
        for field in ["HGCR_relative_to_FIFO", "HGCR_relative_to_MLP", "FIFO_ratio", "GreedyECT_ratio", "Lookahead_ratio", "MLPRanker_ratio"]:
            base[field] = sum(fnum(v.get(field)) for v in vals) / len(vals)
        rows.append(base)
    rows = sorted(rows, key=lambda row: (row["arrival_intensity"], row["carryover_ratio"]))
    labels = [f"{row['arrival_intensity']}-{row['carryover_ratio']}" for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(6.5, len(labels) * 0.75), 5.0), sharex=True)
    axes[0].bar(x - 0.18, [fnum(row["HGCR_relative_to_FIFO"]) * 100.0 for row in rows], width=0.36, label="vs FIFO", color="#2F6BFF")
    axes[0].bar(x + 0.18, [fnum(row["HGCR_relative_to_MLP"]) * 100.0 for row in rows], width=0.36, label="vs MLP", color="#D14B3F")
    axes[0].axhline(0.0, color="#222222", linewidth=0.8)
    axes[0].set_ylabel("Improvement (%)")
    axes[0].legend(frameon=False, ncol=2)
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
    save(fig, out_dir, "fig_G5_action_performance_panel", no_write)
    plt.close(fig)


def fig_g6(data, out_dir, no_write):
    root = Path(data.get("_gantt_dir", GANTT_DIR))
    cases = sorted(root.glob("*/gantt_case1_comparison.png")) if root.exists() else []
    if not cases:
        print("Warning: skip G6, no generated Gantt case comparison found.")
        return
    print(f"Use existing Gantt case for G6: {cases[-1]}")


def eval_history_rows(root: Path) -> List[dict]:
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("*/eval_history*.csv")):
        rows.extend(read_csv(path))
    return rows


def fig_s1(data, out_dir, no_write):
    beta_rows = [row for row in data["summary"] if row.get("method") == "HGCR-PPO" and row.get("arrival_intensity") == "medium" and row.get("carryover_ratio") == "medium"]
    if not beta_rows:
        print("Warning: skip S1, missing beta sensitivity rows.")
        return
    plt = mpl()
    rows = sorted(beta_rows, key=lambda r: fnum(r.get("reward_beta", 0.0)))
    x = list(range(len(rows)))
    cmax = [fnum(r.get("Cmax_mean")) for r in rows]
    # Improvement is unavailable in method summary; show normalized improvement against FIFO from matching rows when present in action table.
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(x, cmax, marker="o", label="HGCR Cmax")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r.get("reward_beta", "")) for r in rows])
    ax.set_xlabel("reward_beta")
    ax.set_ylabel("Cmax mean")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out_dir, "fig_S1_reward_beta_sensitivity", no_write)
    plt.close(fig)


def fig_s2_s3(data, out_dir, no_write):
    rows = eval_history_rows(Path(data.get("_hgcr_runs_dir", HGCR_RUNS_DIR)))
    if not rows:
        print("Warning: skip S2/S3, missing eval_history rows.")
        return
    import numpy as np

    plt = mpl()
    for metric, stem, ylabel in [
        ("eval_Cmax_mean", "fig_S2_training_convergence_cmax", "Eval Cmax mean"),
        ("eval_reward_mean", "fig_S3_training_convergence_reward", "Eval reward mean"),
    ]:
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        by_seed = {}
        for row in rows:
            by_seed.setdefault(str(row.get("seed")), []).append(row)
        max_len = 0
        curves = []
        for seed, vals in sorted(by_seed.items()):
            vals = sorted(vals, key=lambda r: int(float(r.get("episode", 0))))
            x = [int(float(r.get("episode", 0))) for r in vals]
            y = [fnum(r.get(metric)) for r in vals]
            max_len = max(max_len, len(y))
            curves.append(y)
            ax.plot(x, y, marker="o", linewidth=1.0, alpha=0.75, label=f"seed{seed}")
        if curves:
            min_len = min(len(c) for c in curves)
            avg = np.mean([c[:min_len] for c in curves], axis=0)
            x = [int(float(r.get("episode", 0))) for r in sorted(by_seed[sorted(by_seed)[0]], key=lambda r: int(float(r.get("episode", 0))))[:min_len]]
            ax.plot(x, avg, color="#111111", linewidth=2.0, label="mean")
        if metric == "eval_Cmax_mean":
            fifo = [fnum(r.get("baseline_FIFO_Cmax")) for r in rows if r.get("baseline_FIFO_Cmax")]
            if fifo:
                ax.axhline(sum(fifo) / len(fifo), color="#888888", linestyle="--", label="FIFO")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        save(fig, out_dir, stem, no_write)
        plt.close(fig)


def fig_s4(data, out_dir, no_write):
    rows = data.get("arpd", [])
    if not rows:
        print("Warning: skip S4, missing ARPD summary.")
        return
    plt = mpl()
    rows = sorted(rows, key=lambda r: fnum(r.get("ARPD_mean")))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar([r["method"] for r in rows], [fnum(r.get("ARPD_mean")) for r in rows], color="#2F6BFF")
    ax.set_ylabel("ARPD mean (%)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save(fig, out_dir, "fig_S4_method_rank_or_arpd", no_write)
    plt.close(fig)


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
    data["_hgcr_runs_dir"] = args.hgcr_runs_dir
    data["_gantt_dir"] = args.gantt_dir
    for fn in [fig_g1, fig_g2, fig_g3, fig_g4, fig_g5, fig_g6, fig_s1, fig_s2_s3, fig_s4]:
        fn(data, out_dir, no_write)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=str(INPUT_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--hgcr_runs_dir", default=str(HGCR_RUNS_DIR))
    parser.add_argument("--gantt_dir", default=str(GANTT_DIR))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
