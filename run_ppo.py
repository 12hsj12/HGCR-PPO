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
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overfit_one_instance", action="store_true")
    parser.add_argument("--instance_index", type=int, default=0)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args()
    config = PPOConfig(
        size=args.size,
        episodes=args.episodes,
        seed=args.seed,
        eval_interval=args.eval_interval,
        overfit_one_instance=args.overfit_one_instance,
        instance_index=args.instance_index,
        overfit_split=args.split,
    )
    train_ppo, debug_ppo = _load_train_ppo()
    if args.debug:
        debug_ppo(config)
    else:
        train_ppo(config)


if __name__ == "__main__":
    main()
