"""Batch runner for Stage G no-FIFO, no-unknown ablation jobs.

This script prepares the reward-component and action-library ablation runs used
by the Chinese manuscript. It does not evaluate external FIFO baselines; the
internal arrival-order rule is named through the paper-facing display label.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


OUTPUT_DIR = Path("data/results/stage_G/ablation")
SEEDS = [0, 1, 2]


@dataclass(frozen=True)
class AblationJob:
    family: str
    name: str
    reward_mode: str
    disabled_actions: tuple[str, ...] = ()


REWARD_COMPONENT_JOBS = [
    AblationJob("reward_component", "util_only", "util_only"),
    AblationJob("reward_component", "cmax_only", "cmax_only"),
    AblationJob("reward_component", "util_plus_cmax", "util_plus_cmax"),
]

ACTION_LIBRARY_JOBS = [
    AblationJob("action_library", "full", "util_plus_cmax"),
    AblationJob("action_library", "without_arrival_order_rule", "util_plus_cmax", ("Arrival-order rule",)),
    AblationJob("action_library", "without_greedyect_rule", "util_plus_cmax", ("GreedyECT",)),
    AblationJob("action_library", "without_lookahead_rule", "util_plus_cmax", ("Lookahead",)),
    AblationJob("action_library", "without_mlp_ranker_rule", "util_plus_cmax", ("MLP-Ranker",)),
]


def selected_jobs(which: str) -> List[AblationJob]:
    if which == "reward":
        return list(REWARD_COMPONENT_JOBS)
    if which == "action":
        return list(ACTION_LIBRARY_JOBS)
    return [*REWARD_COMPONENT_JOBS, *ACTION_LIBRARY_JOBS]


def build_command(args, job: AblationJob, seed: int) -> List[str]:
    cmd = [
        sys.executable,
        "run_hgcr_dynamic_ppo.py",
        "--size",
        "small",
        "--arrival_intensity",
        "medium",
        "--carryover_ratio",
        "medium",
        "--top_k",
        str(args.top_k),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(seed),
        "--reward_mode",
        job.reward_mode,
        "--reward_beta",
        str(args.reward_beta),
        "--baseline_method",
        "mlp_ranker_soft_ce",
        "--ablation_family",
        job.family,
        "--ablation_name",
        job.name,
        "--output_dir",
        str(args.output_dir),
        "--eval_interval",
        str(args.eval_interval),
        "--disable_early_stop",
        "--no_fifo_outputs",
        "--device",
        args.device,
    ]
    if args.num_scenarios is not None:
        cmd.extend(["--num_scenarios", str(args.num_scenarios)])
    if args.eval_scenarios is not None:
        cmd.extend(["--eval_scenarios", str(args.eval_scenarios)])
    if args.ranker_ckpt:
        cmd.extend(["--ranker_ckpt", args.ranker_ckpt])
    if args.smoke_test:
        cmd.append("--smoke_test")
    if job.disabled_actions:
        cmd.append("--disabled_actions")
        cmd.extend(job.disabled_actions)
    return cmd


def print_command(cmd: Iterable[str]) -> None:
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "reward", "action"], default="all")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--reward_beta", type=float, default=5.0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--num_scenarios", type=int, default=None)
    parser.add_argument("--eval_scenarios", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ranker_ckpt", default="checkpoints/stage_C/mlp_ranker/small_topk5_soft_ce/best.pt")
    args = parser.parse_args()

    jobs = selected_jobs(args.only)
    commands = [build_command(args, job, seed) for job in jobs for seed in args.seeds]
    print(f"Planned Stage G ablation jobs: {len(commands)}")
    for cmd in commands:
        print_command(cmd)
    if args.dry_run:
        print("Dry run enabled: no training commands were executed.")
        return

    for cmd in commands:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
