"""Actor-critic networks for vector-state PPO."""

from __future__ import annotations

import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        max_jobs: int | None = None,
        max_split: int | None = None,
        action_mode: str = "two_head",
    ):
        super().__init__()
        self.action_mode = action_mode
        self.max_jobs = max_jobs
        self.max_split = max_split
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.job_head = nn.Linear(hidden_dim, max_jobs or action_dim)
        self.split_head = nn.Linear(hidden_dim, max_split or 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        value = self.value_head(features).squeeze(-1)
        if self.action_mode == "two_head":
            return self.job_head(features), self.split_head(features), value
        return self.policy_head(features), value

    def masked_distribution(self, obs: torch.Tensor, action_mask: torch.Tensor):
        logits, value = self.forward(obs)
        masked_logits = logits.masked_fill(~action_mask.bool(), -1e9)
        return torch.distributions.Categorical(logits=masked_logits), value

    def masked_two_head_distributions(
        self,
        obs: torch.Tensor,
        job_mask: torch.Tensor,
        selected_job: torch.Tensor | None = None,
        split_masks: torch.Tensor | None = None,
    ):
        job_logits, split_logits, value = self.forward(obs)
        masked_job_logits = job_logits.masked_fill(~job_mask.bool(), -1e9)
        job_dist = torch.distributions.Categorical(logits=masked_job_logits)

        if split_masks is None:
            split_mask = torch.ones_like(split_logits, dtype=torch.bool)
        elif selected_job is None:
            split_mask = split_masks.any(dim=1)
        else:
            split_mask = split_masks[torch.arange(obs.shape[0], device=obs.device), selected_job]
        masked_split_logits = split_logits.masked_fill(~split_mask.bool(), -1e9)
        split_dist = torch.distributions.Categorical(logits=masked_split_logits)
        return job_dist, split_dist, value
