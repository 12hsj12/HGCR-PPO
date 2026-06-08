"""Reward-beta ablation launcher for HGCR dynamic PPO.

By default this script only prints the commands. Pass --execute to run them.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


BETAS = [0.005, 0.01, 0.05]


def build_command(args, beta: float) -> list[str]:
    return [
        sys.executable,
        "run_hgcr_dynamic_ppo.py",
        "--size",
        args.size,
        "--top_k",
        str(args.top_k),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--arrival_intensity",
        args.arrival_intensity,
        "--carryover_ratio",
        args.carryover_ratio,
        "--reward_mode",
        "util_plus_cmax",
        "--reward_beta",
        str(beta),
        "--baseline_method",
        args.baseline_method,
        "--ranker_ckpt",
        args.ranker_ckpt,
        "--learning_rate",
        str(args.learning_rate),
        "--gamma",
        str(args.gamma),
        "--gae_lambda",
        str(args.gae_lambda),
        "--clip_ratio",
        str(args.clip_ratio),
        "--entropy_coef",
        str(args.entropy_coef),
        "--batch_size",
        str(args.batch_size),
        "--mini_batch_size",
        str(args.mini_batch_size),
        "--update_epochs",
        str(args.update_epochs),
        "--eval_interval",
        str(args.eval_interval),
        "--early_stop_patience",
        str(args.early_stop_patience),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrival_intensity", default="medium")
    parser.add_argument("--carryover_ratio", default="medium")
    parser.add_argument("--baseline_method", choices=["fifo", "mlp_ranker_soft_ce"], default="fifo")
    parser.add_argument("--ranker_ckpt", default="checkpoints/stage_C/mlp_ranker/small_topk5_soft_ce/best.pt")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--summary_csv", default="data/results/stage_F/hgcr_dynamic_ppo/reward_beta_ablation_commands.csv")
    args = parser.parse_args()

    rows = []
    for beta in BETAS:
        cmd = build_command(args, beta)
        rows.append({"reward_beta": beta, "command": " ".join(cmd)})
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, check=True)

    out = Path(args.summary_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["reward_beta", "command"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved ablation command summary to {out}")


if __name__ == "__main__":
    main()
