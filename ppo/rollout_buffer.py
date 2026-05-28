"""Rollout buffer with GAE advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    returns: torch.Tensor
    raw_returns: torch.Tensor
    advantages: torch.Tensor
    old_values: torch.Tensor
    flat_masks: torch.Tensor
    job_masks: torch.Tensor
    split_masks: torch.Tensor
    stats: Dict[str, float]


@dataclass
class RolloutBuffer:
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    flat_masks: List[np.ndarray] = field(default_factory=list)
    job_masks: List[np.ndarray] = field(default_factory=list)
    split_masks: List[np.ndarray] = field(default_factory=list)

    def add(self, obs, action, logprob, reward, done, value, masks) -> None:
        self.observations.append(obs.copy())
        self.actions.append(np.array(action, dtype=np.int64).reshape(-1))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.flat_masks.append(masks["flat"].copy())
        self.job_masks.append(masks["job"].copy())
        self.split_masks.append(masks["split"].copy())

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.flat_masks.clear()
        self.job_masks.clear()
        self.split_masks.clear()

    def compute_returns_advantages(
        self,
        gamma: float,
        gae_lambda: float,
        use_return_normalization: bool = True,
        last_value: float = 0.0,
    ) -> RolloutBatch:
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[t] = gae

        raw_returns = advantages + values[:-1]
        adv_mean = float(advantages.mean()) if len(advantages) else 0.0
        adv_std = float(advantages.std()) if len(advantages) else 0.0
        norm_advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        ret_mean = float(raw_returns.mean()) if len(raw_returns) else 0.0
        ret_std = float(raw_returns.std()) if len(raw_returns) else 0.0
        returns = raw_returns.copy()
        if use_return_normalization:
            returns = (returns - ret_mean) / (ret_std + 1e-8)

        actions = np.array(self.actions, dtype=np.int64)
        if actions.ndim == 1:
            actions = actions[:, None]

        stats = {
            "advantage_mean": adv_mean,
            "advantage_std": adv_std,
            "return_mean": ret_mean,
            "return_std": ret_std,
            "value_pred_mean": float(np.array(self.values, dtype=np.float32).mean()) if self.values else 0.0,
            "value_target_mean": float(returns.mean()) if len(returns) else 0.0,
            "raw_value_target_mean": ret_mean,
        }

        return RolloutBatch(
            observations=torch.tensor(np.array(self.observations), dtype=torch.float32),
            actions=torch.tensor(actions, dtype=torch.long),
            old_logprobs=torch.tensor(self.logprobs, dtype=torch.float32),
            returns=torch.tensor(returns, dtype=torch.float32),
            raw_returns=torch.tensor(raw_returns, dtype=torch.float32),
            advantages=torch.tensor(norm_advantages, dtype=torch.float32),
            old_values=torch.tensor(self.values, dtype=torch.float32),
            flat_masks=torch.tensor(np.array(self.flat_masks), dtype=torch.bool),
            job_masks=torch.tensor(np.array(self.job_masks), dtype=torch.bool),
            split_masks=torch.tensor(np.array(self.split_masks), dtype=torch.bool),
            stats=stats,
        )
