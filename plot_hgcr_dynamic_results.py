"""Plot HGCR dynamic PPO training and evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _first(pattern: str, root: Path) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def _require_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def plot_reward_curve(run_dir: Path, output_dir: Path) -> Path | None:
    curve_path = _first("reward_cmax_curve__*.csv", run_dir)
    if curve_path is None:
        return None
    rows = _read_csv(curve_path)
    if not rows:
        return None
    plt = _require_matplotlib()
    episodes = [int(row["episode"]) for row in rows]
    rewards = [float(row["total_reward"]) for row in rows]
    out = output_dir / "reward_curve.png"
    plt.figure(figsize=(8, 4.5))
    plt.plot(episodes, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_cmax_curve(run_dir: Path, output_dir: Path) -> Path | None:
    curve_path = _first("reward_cmax_curve__*.csv", run_dir)
    if curve_path is None:
        return None
    rows = _read_csv(curve_path)
    if not rows:
        return None
    plt = _require_matplotlib()
    episodes = [int(row["episode"]) for row in rows]
    cmax = [float(row["episode_Cmax"]) for row in rows]
    out = output_dir / "cmax_curve.png"
    plt.figure(figsize=(8, 4.5))
    plt.plot(episodes, cmax)
    plt.xlabel("Episode")
    plt.ylabel("Episode Cmax")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_action_ratio(run_dir: Path, output_dir: Path) -> Path | None:
    path = _first("action_ratio__*.csv", run_dir)
    if path is None:
        return None
    rows = _read_csv(path)
    if not rows:
        return None
    plt = _require_matplotlib()
    names = [row["rule_name"] for row in rows]
    ratios = [float(row["selection_ratio"]) for row in rows]
    out = output_dir / "action_ratio_bar.png"
    plt.figure(figsize=(8, 4.5))
    plt.bar(names, ratios)
    plt.ylabel("Selection ratio")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_method_comparison(run_dir: Path, output_dir: Path) -> Path | None:
    path = _first("eval_summary__*.csv", run_dir)
    if path is None:
        return None
    rows = _read_csv(path)
    if not rows:
        return None
    plt = _require_matplotlib()
    methods = [row["method"] for row in rows]
    cmax = [float(row["Cmax_mean"]) for row in rows]
    out = output_dir / "method_comparison_bar.png"
    plt.figure(figsize=(8, 4.5))
    plt.bar(methods, cmax)
    plt.ylabel("Cmax mean")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def _run_beta(run_dir: Path) -> float | None:
    path = _first("manifest__*.json", run_dir)
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        return float(data["args"]["reward_beta"])
    except (KeyError, TypeError, ValueError):
        return None


def plot_beta_sensitivity(runs_root: Path, output_dir: Path) -> Path | None:
    points = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        beta = _run_beta(run_dir)
        summary_path = _first("eval_summary__*.csv", run_dir)
        if beta is None or summary_path is None:
            continue
        rows = _read_csv(summary_path)
        hgcr = [row for row in rows if row.get("method") == "HGCR-PPO"]
        if hgcr:
            points.append((beta, float(hgcr[0]["Cmax_mean"])))
    if len(points) < 2:
        return None
    points.sort()
    plt = _require_matplotlib()
    out = output_dir / "reward_beta_sensitivity.png"
    plt.figure(figsize=(7, 4.5))
    plt.plot([p[0] for p in points], [p[1] for p in points], marker="o")
    plt.xlabel("Reward beta")
    plt.ylabel("HGCR-PPO Cmax mean")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def try_plot_gantt(run_dir: Path, output_dir: Path) -> Path | None:
    marker = run_dir / "gantt_fifo_vs_hgcr.source"
    if not marker.exists():
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--runs_root", default="data/results/stage_F/hgcr_dynamic_ppo/runs")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir()) if runs_root.exists() else []
        if not run_dirs:
            raise FileNotFoundError(f"No run directories found under {runs_root}")
        run_dir = run_dirs[-1]
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Path | None] = {
        "reward_curve": plot_reward_curve(run_dir, output_dir),
        "cmax_curve": plot_cmax_curve(run_dir, output_dir),
        "action_ratio_bar": plot_action_ratio(run_dir, output_dir),
        "method_comparison_bar": plot_method_comparison(run_dir, output_dir),
        "reward_beta_sensitivity": plot_beta_sensitivity(runs_root, output_dir),
        "gantt_fifo_vs_hgcr": try_plot_gantt(run_dir, output_dir),
    }
    for name, path in outputs.items():
        if path is not None:
            print(f"{name}: {path}")
        else:
            print(f"{name}: skipped")


if __name__ == "__main__":
    main()
