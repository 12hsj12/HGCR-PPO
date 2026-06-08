"""Official HGCR-PPO training entry for Stage F.

Static Stage F runs keep using the historical rule-library PPO path. Dynamic
DRL-style runs are delegated to ``run_hgcr_dynamic_ppo.py``.
"""

from __future__ import annotations

import argparse
import sys

from instance_manager import SIZES, SPLITS
from run_rule_library_ppo import run


def _dynamic_requested(argv: list[str]) -> bool:
    for idx, item in enumerate(argv):
        if item == "--scenario_mode" and idx + 1 < len(argv) and argv[idx + 1] == "dynamic":
            return True
        if item == "--scenario_mode=dynamic":
            return True
    return False


def _forward_without_scenario_mode(argv: list[str]) -> list[str]:
    forwarded = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item == "--scenario_mode":
            skip_next = True
            continue
        if item.startswith("--scenario_mode="):
            continue
        forwarded.append(item)
    return forwarded


def main() -> None:
    if _dynamic_requested(sys.argv[1:]):
        from run_hgcr_dynamic_ppo import main as dynamic_main

        sys.argv = [sys.argv[0], *_forward_without_scenario_mode(sys.argv[1:])]
        dynamic_main()
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_mode", choices=["static", "dynamic"], default="static")
    parser.add_argument("--size", choices=SIZES, default="small")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ranker_ckpt", default="checkpoints/stage_C/mlp_ranker/small_topk5_soft_ce/best.pt")
    parser.add_argument(
        "--bc_init_ckpt",
        default=r"checkpoints\stage_F\rule_library_bc\RuleLibBC_small_k5_ep50_20260606-095923_7ee6cc0c\best.pt",
    )
    parser.add_argument("--baseline_method", choices=["mlp_ranker_soft_ce"], default="mlp_ranker_soft_ce")
    parser.add_argument("--reward_mode", choices=["step_plus_final_delta", "conservative_final_delta"], default="step_plus_final_delta")
    parser.add_argument("--use_rule_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze_actor_episodes", type=int, default=20)
    parser.add_argument("--kl_to_bc_coef", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--eval_interval", type=int, default=25)
    parser.add_argument("--early_stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--final_reward_weight", type=float, default=1.0)
    parser.add_argument("--step_reward_weight", type=float, default=1.0)
    parser.add_argument("--override_penalty", type=float, default=0.03)
    parser.add_argument("--keep_base_bias", type=float, default=2.0)
    parser.add_argument("--output_dir", default="data/results/stage_F/hgcr_ppo")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--update_epochs", type=int, default=4)
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    args.method = "HGCR-PPO"
    args.disable_min_cmax_action = True
    run(args)


if __name__ == "__main__":
    main()
