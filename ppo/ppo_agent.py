"""Hand-written PPO agent with flattened and two-head action modes."""

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
        self.model = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            max_jobs=config.limits["max_jobs"],
            max_split=config.limits["max_split"],
            action_mode=config.action_mode,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def select_action(self, obs: np.ndarray, masks, greedy: bool = False) -> Tuple[np.ndarray | int, float, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.config.policy_mode == "order_only":
                job_mask = torch.tensor(masks["job"], dtype=torch.bool, device=self.device).unsqueeze(0)
                job_logits, _, value = self.model(obs_t)
                job_dist = torch.distributions.Categorical(logits=job_logits.masked_fill(~job_mask, -1e9))
                job_action = torch.argmax(job_dist.probs, dim=-1) if greedy else job_dist.sample()
                logprob = job_dist.log_prob(job_action)
                action = np.array([int(job_action.item()), 0], dtype=np.int64)
                return action, float(logprob.item()), float(value.item())

            if self.config.action_mode == "two_head":
                job_mask = torch.tensor(masks["job"], dtype=torch.bool, device=self.device).unsqueeze(0)
                split_masks = torch.tensor(masks["split"], dtype=torch.bool, device=self.device).unsqueeze(0)
                job_dist, _, value = self.model.masked_two_head_distributions(obs_t, job_mask, split_masks=split_masks)
                job_action = torch.argmax(job_dist.probs, dim=-1) if greedy else job_dist.sample()
                _, split_dist, _ = self.model.masked_two_head_distributions(
                    obs_t,
                    job_mask,
                    selected_job=job_action,
                    split_masks=split_masks,
                )
                split_action = torch.argmax(split_dist.probs, dim=-1) if greedy else split_dist.sample()
                logprob = job_dist.log_prob(job_action) + split_dist.log_prob(split_action)
                action = np.array([int(job_action.item()), int(split_action.item())], dtype=np.int64)
                return action, float(logprob.item()), float(value.item())

            mask_t = torch.tensor(masks["flat"] if isinstance(masks, dict) else masks, dtype=torch.bool, device=self.device).unsqueeze(0)
            dist, value = self.model.masked_distribution(obs_t, mask_t)
            action_t = torch.argmax(dist.probs, dim=-1) if greedy else dist.sample()
            logprob = dist.log_prob(action_t)
            return int(action_t.item()), float(logprob.item()), float(value.item())

    def update(self, buffer: RolloutBuffer, episode: int | None = None) -> Dict[str, float]:
        if not buffer.rewards:
            return self._zero_stats()

        batch = buffer.compute_returns_advantages(
            self.config.gamma,
            self.config.gae_lambda,
            use_return_normalization=self.config.use_return_normalization,
        )
        obs = batch.observations.to(self.device)
        actions = batch.actions.to(self.device)
        old_logprobs = batch.old_logprobs.to(self.device)
        returns = batch.returns.to(self.device)
        advantages = batch.advantages.to(self.device)
        old_values = batch.old_values.to(self.device)
        flat_masks = batch.flat_masks.to(self.device)
        job_masks = batch.job_masks.to(self.device)
        split_masks = batch.split_masks.to(self.device)

        n = len(actions)
        batch_size = min(self.config.minibatch_size, n)
        stats = self._zero_stats()
        update_count = 0

        for _ in range(self.config.update_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                mb_idx = indices[start : start + batch_size]
                if self.config.action_mode == "two_head":
                    if self.config.policy_mode == "order_only":
                        logprobs, values, entropy, job_entropy, split_entropy = self._order_only_eval(
                            obs[mb_idx],
                            actions[mb_idx],
                            job_masks[mb_idx],
                        )
                    else:
                        logprobs, values, entropy, job_entropy, split_entropy = self._two_head_eval(
                            obs[mb_idx],
                            actions[mb_idx],
                            job_masks[mb_idx],
                            split_masks[mb_idx],
                        )
                else:
                    dist, values = self.model.masked_distribution(obs[mb_idx], flat_masks[mb_idx])
                    action_ids = actions[mb_idx, 0]
                    logprobs = dist.log_prob(action_ids)
                    entropy = dist.entropy().mean()
                    job_entropy = torch.zeros((), device=self.device)
                    split_entropy = torch.zeros((), device=self.device)

                ratio = torch.exp(logprobs - old_logprobs[mb_idx])
                unclipped = ratio * advantages[mb_idx]
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages[mb_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = self._value_loss(values, returns[mb_idx], old_values[mb_idx])

                if self.config.action_mode == "two_head":
                    job_coef, split_coef = self._entropy_coefs(episode)
                    if self.config.policy_mode == "order_only":
                        entropy_bonus = job_coef * job_entropy
                    else:
                        entropy_bonus = job_coef * job_entropy + split_coef * split_entropy
                else:
                    entropy_bonus = self._entropy_coef(episode) * entropy
                loss = policy_loss + self.config.value_coef * value_loss - entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_logprobs[mb_idx] - logprobs).mean()
                    clip_fraction = ((ratio < 1.0 - self.config.clip_ratio) | (ratio > 1.0 + self.config.clip_ratio)).float().mean()
                    explained_variance = self._explained_variance(values, returns[mb_idx])

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())
                stats["total_entropy"] += float(entropy.item())
                stats["job_entropy"] += float(job_entropy.item())
                stats["split_entropy"] += float(split_entropy.item())
                stats["approx_kl"] += float(approx_kl.item())
                stats["clip_fraction"] += float(clip_fraction.item())
                stats["explained_variance"] += float(explained_variance)
                stats["grad_norm"] += float(grad_norm)
                stats["value_pred_mean"] += float(values.mean().item())
                stats["value_target_mean"] += float(returns[mb_idx].mean().item())
                update_count += 1

        buffer.clear()
        averaged = {key: value / max(1, update_count) for key, value in stats.items()}
        averaged.update(
            {
                "advantage_mean": batch.stats["advantage_mean"],
                "advantage_std": batch.stats["advantage_std"],
                "return_mean": batch.stats["return_mean"],
                "return_std": batch.stats["return_std"],
                "raw_value_target_mean": batch.stats["raw_value_target_mean"],
            }
        )
        return averaged

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

    def freeze_actor_policy(self) -> None:
        """Freeze policy-producing modules and shared encoder for BC-policy diagnostics."""

        for module in [self.model.shared, self.model.policy_head, self.model.job_head, self.model.split_head]:
            for parameter in module.parameters():
                parameter.requires_grad = False
        for parameter in self.model.value_head.parameters():
            parameter.requires_grad = True

    def action_debug_stats(self, obs: np.ndarray, masks, action) -> Dict[str, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.config.policy_mode == "order_only":
                job_mask = torch.tensor(masks["job"], dtype=torch.bool, device=self.device).unsqueeze(0)
                job_logits, _, _ = self.model(obs_t)
                job_dist = torch.distributions.Categorical(logits=job_logits.masked_fill(~job_mask, -1e9))
                return {"job_entropy": float(job_dist.entropy().item()), "split_entropy": 0.0}

            if self.config.action_mode == "two_head":
                job_mask = torch.tensor(masks["job"], dtype=torch.bool, device=self.device).unsqueeze(0)
                split_masks = torch.tensor(masks["split"], dtype=torch.bool, device=self.device).unsqueeze(0)
                action_t = torch.tensor([int(action[0])], dtype=torch.long, device=self.device)
                job_dist, split_dist, _ = self.model.masked_two_head_distributions(
                    obs_t,
                    job_mask,
                    selected_job=action_t,
                    split_masks=split_masks,
                )
                return {
                    "job_entropy": float(job_dist.entropy().item()),
                    "split_entropy": float(split_dist.entropy().item()),
                }
            flat_mask = torch.tensor(masks["flat"], dtype=torch.bool, device=self.device).unsqueeze(0)
            dist, _ = self.model.masked_distribution(obs_t, flat_mask)
            return {"job_entropy": 0.0, "split_entropy": float(dist.entropy().item())}

    def _two_head_eval(self, obs, actions, job_masks, split_masks):
        job_actions = actions[:, 0]
        split_actions = actions[:, 1]
        job_dist, split_dist, values = self.model.masked_two_head_distributions(
            obs,
            job_masks,
            selected_job=job_actions,
            split_masks=split_masks,
        )
        logprobs = job_dist.log_prob(job_actions) + split_dist.log_prob(split_actions)
        job_entropy = job_dist.entropy().mean()
        split_entropy = split_dist.entropy().mean()
        return logprobs, values, job_entropy + split_entropy, job_entropy, split_entropy

    def _order_only_eval(self, obs, actions, job_masks):
        job_actions = actions[:, 0]
        job_logits, _, values = self.model(obs)
        job_dist = torch.distributions.Categorical(logits=job_logits.masked_fill(~job_masks.bool(), -1e9))
        logprobs = job_dist.log_prob(job_actions)
        job_entropy = job_dist.entropy().mean()
        split_entropy = torch.zeros((), device=self.device)
        return logprobs, values, job_entropy, job_entropy, split_entropy

    def behavior_clone_job_head(self, expert_samples, epochs: int = 20, batch_size: int | None = None) -> Dict[str, float]:
        if not expert_samples or epochs <= 0:
            return {"bc_loss": 0.0, "bc_accuracy": 0.0}

        obs = torch.tensor(np.array([sample["observation"] for sample in expert_samples]), dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array([sample["job_mask"] for sample in expert_samples]), dtype=torch.bool, device=self.device)
        targets = torch.tensor([sample["expert_job_slot"] for sample in expert_samples], dtype=torch.long, device=self.device)
        n = len(targets)
        batch_size = min(batch_size or self.config.minibatch_size, n)
        last_loss = 0.0
        last_acc = 0.0

        self.train()
        for _ in range(epochs):
            indices = torch.randperm(n, device=self.device)
            correct = 0
            total = 0
            loss_sum = 0.0
            for start in range(0, n, batch_size):
                idx = indices[start : start + batch_size]
                job_logits, _, _ = self.model(obs[idx])
                logits = job_logits.masked_fill(~masks[idx], -1e9)
                loss = nn.functional.cross_entropy(logits, targets[idx])
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    pred = torch.argmax(logits, dim=-1)
                    correct += int((pred == targets[idx]).sum().item())
                    total += int(len(idx))
                    loss_sum += float(loss.item()) * len(idx)
            last_loss = loss_sum / max(1, total)
            last_acc = correct / max(1, total)
        return {"bc_loss": last_loss, "bc_accuracy": last_acc}

    def _value_loss(self, values, returns, old_values):
        value_pred = values
        if self.config.use_value_clip:
            value_pred_clipped = old_values + torch.clamp(
                values - old_values,
                -self.config.value_clip_range,
                self.config.value_clip_range,
            )
            loss_unclipped = self._base_value_loss(value_pred, returns)
            loss_clipped = self._base_value_loss(value_pred_clipped, returns)
            return torch.max(loss_unclipped, loss_clipped)
        return self._base_value_loss(value_pred, returns)

    def _base_value_loss(self, values, returns):
        if self.config.value_loss_type == "mse":
            return nn.functional.mse_loss(values, returns)
        return nn.functional.smooth_l1_loss(values, returns)

    def _entropy_coef(self, episode: int | None) -> float:
        if not self.config.use_entropy_annealing or episode is None:
            return self.config.entropy_coef
        frac = min(1.0, max(0.0, episode / max(1, self.config.entropy_anneal_episodes)))
        return self.config.entropy_coef_start + frac * (self.config.entropy_coef_end - self.config.entropy_coef_start)

    def _entropy_coefs(self, episode: int | None) -> tuple[float, float]:
        if not self.config.use_entropy_annealing or episode is None:
            return self.config.job_entropy_coef, self.config.split_entropy_coef
        base = self._entropy_coef(episode)
        start = max(self.config.entropy_coef_start, 1e-8)
        scale = base / start
        return self.config.job_entropy_coef * scale, self.config.split_entropy_coef * scale

    @staticmethod
    def _explained_variance(values, returns) -> float:
        target_var = torch.var(returns)
        if target_var.item() < 1e-8:
            return 0.0
        return float((1.0 - torch.var(returns - values) / (target_var + 1e-8)).item())

    @staticmethod
    def _zero_stats() -> Dict[str, float]:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_entropy": 0.0,
            "job_entropy": 0.0,
            "split_entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "explained_variance": 0.0,
            "grad_norm": 0.0,
            "advantage_mean": 0.0,
            "advantage_std": 0.0,
            "return_mean": 0.0,
            "return_std": 0.0,
            "value_pred_mean": 0.0,
            "value_target_mean": 0.0,
            "raw_value_target_mean": 0.0,
        }
