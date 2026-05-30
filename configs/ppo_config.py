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
    entropy_coef: float = 0.05
    job_entropy_coef: float = 0.03
    split_entropy_coef: float = 0.08
    value_coef: float = 0.25
    max_grad_norm: float = 0.5
    update_epochs: int = 5
    minibatch_size: int = 64
    hidden_dim: int = 256
    alpha_final_reward: float = 0.01
    eval_interval: int = 10
    action_mode: str = "two_head"
    policy_mode: str = "order_split"
    split_rule: str = "greedy_ect"
    bc_pretrain: bool = False
    bc_epochs: int = 20
    expert_heuristic: str = "GreedyECT"
    regenerate_expert: bool = False
    reward_mode: str = "normalized_delta_plus_baseline_final"
    reward_scale: float = 1.0
    reward_clip: float = 1.0
    final_reward_beta: float = 1.0
    illegal_action_penalty: float = -0.1
    reference_baseline: str = "FIFO"
    use_reward_normalization: bool = True
    use_return_normalization: bool = True
    value_loss_type: str = "huber"
    use_value_clip: bool = True
    value_clip_range: float = 0.2
    use_entropy_annealing: bool = True
    entropy_coef_start: float = 0.08
    entropy_coef_end: float = 0.01
    entropy_anneal_episodes: int = 500
    waiting_time_penalty: float = 0.0
    load_balance_penalty: float = 0.0
    excessive_split_penalty: float = 0.0
    utilization_bonus: float = 0.0
    train_split: str = "train"
    test_split: str = "test"
    overfit_one_instance: bool = False
    instance_index: int = 0
    overfit_split: str = "train"

    @property
    def limits(self) -> dict:
        return SIZE_LIMITS[self.size]
