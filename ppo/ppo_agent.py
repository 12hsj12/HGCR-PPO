"""Hand-written PPO agent."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from configs.ppo_config import PPOConfig
from ppo.networks import ActorCritic
from ppo.rollout_buffer import RolloutBuffer


class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig, device: str | None = None):
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ActorCritic(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def select_action(self, obs: np.ndarray, mask: np.ndarray, greedy: bool = False) -> Tuple[int, float, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.model.masked_distribution(obs_t, mask_t)
            if greedy:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.sample()
            logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        if not buffer.rewards:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        obs, actions, old_logprobs, returns, advantages, masks = buffer.compute_returns_advantages(
            self.config.gamma, self.config.gae_lambda
        )
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        old_logprobs = old_logprobs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        masks = masks.to(self.device)

        n = len(actions)
        batch_size = min(self.config.minibatch_size, n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        update_count = 0

        for _ in range(self.config.update_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                mb_idx = indices[start : start + batch_size]
                dist, values = self.model.masked_distribution(obs[mb_idx], masks[mb_idx])
                logprobs = dist.log_prob(actions[mb_idx])
                entropy = dist.entropy().mean()
                ratio = torch.exp(logprobs - old_logprobs[mb_idx])
                unclipped = ratio * advantages[mb_idx]
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages[mb_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, returns[mb_idx])
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())
                update_count += 1

        buffer.clear()
        return {key: value / max(1, update_count) for key, value in stats.items()}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict(), "config": self.config.__dict__}, path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()
