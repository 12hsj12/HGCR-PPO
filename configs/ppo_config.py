"""Default PPO experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass


SIZE_LIMITS = {
    "small": {"max_jobs": 30, "max_machines": 8, "max_split": 3, "size_id": 0},
    "medium": {"max_jobs": 80, "max_machines": 12, "max_split": 4, "size_id": 1},
    "large": {"max_jobs": 150, "max_machines": 18, "max_split": 4, "size_id": 2},
}


@dataclass
class PPOConfig:
    size: str = "small"
    episodes: int = 100
    seed: int = 42
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 5
    minibatch_size: int = 64
    hidden_dim: int = 256
    alpha_final_reward: float = 0.01
    illegal_action_penalty: float = -10.0
    eval_interval: int = 20
    waiting_time_penalty: float = 0.0
    load_balance_penalty: float = 0.0
    excessive_split_penalty: float = 0.0
    utilization_bonus: float = 0.0
    train_split: str = "train"
    test_split: str = "test"

    @property
    def limits(self) -> dict:
        return SIZE_LIMITS[self.size]
