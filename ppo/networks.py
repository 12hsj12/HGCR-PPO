"""Actor-critic networks for vector-state PPO."""

from __future__ import annotations

import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def masked_distribution(self, obs: torch.Tensor, action_mask: torch.Tensor):
        logits, value = self.forward(obs)
        masked_logits = logits.masked_fill(~action_mask.bool(), -1e9)
        return torch.distributions.Categorical(logits=masked_logits), value
