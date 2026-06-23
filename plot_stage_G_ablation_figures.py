"""Plot Stage G ablation summaries when remote results are available."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


PAPER_DATA_DIR = Path("paper_zh/data_used")
FIGURE_DIR = Path("paper_zh/figures/appendix")
REWARD_SUMMARY = "stage_G_reward_component_ablation_summary_no_fifo.csv"
ACTION_SUMMARY = "stage_G_action_library_ablation_summary_no_fifo.csv"


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fnum(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def plot_bar(rows: List[dict], order: List[str], labels: List[str], title: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    by_name = {row["ablation_name"]: row for row in rows if row.get("method") == "HGCR-PPO"}
    values = [fnum(by_name.get(name, {}).get("Cmax_mean_across_seeds")) for name in order]
    errors = [fnum(by_name.get(name, {}).get("Cmax_std_across_seeds")) for name in order]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(labels, values, yerr=errors, color="#4C78A8", edgecolor="#2D3A4A", linewidth=0.8, capsize=3)
    ax.set_ylabel("Mean Cmax across seeds")
    ax.set_title(title)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    for tick in ax.get_xticklabels():
        tick.set_rotation(18)
        tick.set_ha("right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_data_dir", default=str(PAPER_DATA_DIR))
    parser.add_argument("--figure_dir", default=str(FIGURE_DIR))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.paper_data_dir)
    figure_dir = Path(args.figure_dir)
    reward_rows = read_csv(data_dir / REWARD_SUMMARY)
    action_rows = read_csv(data_dir / ACTION_SUMMARY)
    planned = [
        figure_dir / "Fig_A5_reward_component_ablation_no_fifo.png",
        figure_dir / "Fig_A6_action_library_ablation_no_fifo.png",
    ]
    for path in planned:
        print(f"Planned figure: {path}")
    if args.dry_run:
        print("Dry run enabled: no figures were written.")
        return
    if reward_rows:
        plot_bar(
            reward_rows,
            ["util_only", "cmax_only", "util_plus_cmax"],
            ["util only", "Cmax only", "util + Cmax"],
            "Reward component ablation",
            planned[0],
        )
        print(f"Wrote {planned[0]}")
    else:
        print(f"Skip reward plot: missing {data_dir / REWARD_SUMMARY}")
    if action_rows:
        plot_bar(
            action_rows,
            [
                "full",
                "without_arrival_order_rule",
                "without_greedyect_rule",
                "without_lookahead_rule",
                "without_mlp_ranker_rule",
            ],
            ["full", "w/o arrival-order", "w/o GreedyECT", "w/o Lookahead", "w/o MLP-Ranker"],
            "Action library ablation",
            planned[1],
        )
        print(f"Wrote {planned[1]}")
    else:
        print(f"Skip action plot: missing {data_dir / ACTION_SUMMARY}")


if __name__ == "__main__":
    main()
