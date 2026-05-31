"""Run one PPO experiment for a single size and episode budget."""

from __future__ import annotations

import argparse

from configs.ppo_config import PPOConfig


def _load_train_ppo():
    try:
        from ppo.train_ppo import debug_ppo, train_ppo
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "PyTorch is required for PPO training. Install dependencies with: "
                "pip install -r requirements.txt"
            ) from exc
        raise
    return train_ppo, debug_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["small", "medium", "large"], required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overfit_one_instance", action="store_true")
    parser.add_argument("--instance_index", type=int, default=0)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--policy_mode", choices=["order_split", "order_only"], default="order_split")
    parser.add_argument("--split_rule", choices=["min1", "max_feasible", "greedy_ect"], default="greedy_ect")
    parser.add_argument("--bc_pretrain", action="store_true")
    parser.add_argument("--bc_epochs", type=int, default=20)
    parser.add_argument("--expert_heuristic", default="GreedyECT")
    parser.add_argument("--regenerate_expert", action="store_true")
    parser.add_argument("--bc_only_eval", action="store_true")
    parser.add_argument("--freeze_bc_policy", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=PPOConfig.learning_rate)
    parser.add_argument("--clip_ratio", type=float, default=PPOConfig.clip_ratio)
    parser.add_argument("--update_epochs", type=int, default=PPOConfig.update_epochs)
    args = parser.parse_args()
    config = PPOConfig(
        size=args.size,
        episodes=args.episodes,
        seed=args.seed,
        eval_interval=args.eval_interval,
        overfit_one_instance=args.overfit_one_instance,
        instance_index=args.instance_index,
        overfit_split=args.split,
        policy_mode=args.policy_mode,
        split_rule=args.split_rule,
        bc_pretrain=args.bc_pretrain,
        bc_epochs=args.bc_epochs,
        expert_heuristic=args.expert_heuristic,
        regenerate_expert=args.regenerate_expert,
        bc_only_eval=args.bc_only_eval,
        freeze_bc_policy=args.freeze_bc_policy,
        learning_rate=args.learning_rate,
        clip_ratio=args.clip_ratio,
        update_epochs=args.update_epochs,
    )
    print(
        "PPO config: "
        f"learning_rate={config.learning_rate}, "
        f"clip_ratio={config.clip_ratio}, "
        f"update_epochs={config.update_epochs}, "
        f"policy_mode={config.policy_mode}, "
        f"split_rule={config.split_rule}, "
        f"bc_pretrain={config.bc_pretrain}, "
        f"freeze_bc_policy={config.freeze_bc_policy}"
    )
    train_ppo, debug_ppo = _load_train_ppo()
    if args.debug:
        debug_ppo(config)
    else:
        train_ppo(config)


if __name__ == "__main__":
    main()
