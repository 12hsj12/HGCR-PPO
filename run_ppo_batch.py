"""Run PPO sequentially for small, medium, and large at a fixed episode budget."""

from __future__ import annotations

import argparse

from configs.ppo_config import PPOConfig


def _load_train_ppo():
    try:
        from ppo.train_ppo import train_ppo
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "PyTorch is required for PPO training. Install dependencies with: "
                "pip install -r requirements.txt"
            ) from exc
        raise
    return train_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, choices=[100, 500, 1000], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int, default=20)
    args = parser.parse_args()
    train_ppo = _load_train_ppo()
    for size in ["small", "medium", "large"]:
        print(f"\n=== PPO {size} episodes={args.episodes} ===")
        train_ppo(PPOConfig(size=size, episodes=args.episodes, seed=args.seed, eval_interval=args.eval_interval))


if __name__ == "__main__":
    main()
