"""Train Conservative Rule-Selector PPO for HGCR-PPO Stage F."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from instance_manager import SIZES, ensure_fixed_dataset, load_fixed_instances
from rule_selector_env import ALL_RULES, RuleSelectorEnv
from utils.experiment_io import make_result_path, make_run_dir, make_run_id, update_latest_dir, write_csv, progress_iter


@dataclass
class RuleRolloutBuffer:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def add(self, state, action, mask, logprob, reward, done, value) -> None:
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(int(action))
        self.masks.append(np.asarray(mask, dtype=bool).copy())
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.masks.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def as_tensors(self, gamma: float, gae_lambda: float, device: torch.device) -> Dict[str, torch.Tensor]:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray([*self.values, 0.0], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) if len(advantages) else advantages
        return {
            "states": torch.tensor(np.asarray(self.states), dtype=torch.float32, device=device),
            "actions": torch.tensor(self.actions, dtype=torch.long, device=device),
            "masks": torch.tensor(np.asarray(self.masks), dtype=torch.bool, device=device),
            "old_logprobs": torch.tensor(self.logprobs, dtype=torch.float32, device=device),
            "returns": torch.tensor(returns, dtype=torch.float32, device=device),
            "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
        }


class RuleActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(states)
        return self.actor(x), self.critic(x).squeeze(-1)

    def dist_value(self, states: torch.Tensor, masks: torch.Tensor) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        logits, values = self(states)
        masked_logits = logits.masked_fill(~masks.bool(), -1e9)
        return torch.distributions.Categorical(logits=masked_logits), values


def select_action(model: RuleActorCritic, state: np.ndarray, mask: np.ndarray, device: torch.device, greedy: bool = False):
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
    with torch.no_grad():
        dist, value = model.dist_value(state_t, mask_t)
        action_t = torch.argmax(dist.probs, dim=-1) if greedy else dist.sample()
        logprob = dist.log_prob(action_t)
    return int(action_t.item()), float(logprob.item()), float(value.item())


def update_model(
    model: RuleActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RuleRolloutBuffer,
    gamma: float,
    gae_lambda: float,
    clip_ratio: float,
    entropy_coef: float,
    value_coef: float,
    update_epochs: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    if not buffer.rewards:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    batch = buffer.as_tensors(gamma, gae_lambda, device)
    n = len(batch["actions"])
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "grad_norm": 0.0}
    updates = 0
    for _ in range(update_epochs):
        indices = torch.randperm(n, device=device)
        for start in range(0, n, min(batch_size, n)):
            mb = indices[start : start + min(batch_size, n)]
            dist, values = model.dist_value(batch["states"][mb], batch["masks"][mb])
            logprobs = dist.log_prob(batch["actions"][mb])
            ratio = torch.exp(logprobs - batch["old_logprobs"][mb])
            unclipped = ratio * batch["advantages"][mb]
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch["advantages"][mb]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.smooth_l1_loss(values, batch["returns"][mb])
            entropy = dist.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy.item())
            stats["grad_norm"] += float(grad_norm)
            updates += 1
    buffer.clear()
    return {key: value / max(1, updates) for key, value in stats.items()}


def evaluate_validation(model, instances, args, device: torch.device, max_eval_instances: int = 5) -> float:
    cmax_values = []
    for instance in instances[:max_eval_instances]:
        env = RuleSelectorEnv(
            instance,
            top_k=args.top_k,
            mlp_soft_model_path=args.mlp_soft_model_path,
            mlp_pairwise_model_path=args.mlp_pairwise_model_path,
        )
        state = env.reset(instance)
        while not env.env.is_done():
            action, _, _ = select_action(model, state, env.action_mask(), device, greedy=True)
            state, _, _, _ = env.step(action)
        cmax_values.append(float(env.env.current_cmax))
    return mean(cmax_values) if cmax_values else float("inf")


def save_checkpoint(path: Path, model: RuleActorCritic, args, state_dim: int, action_dim: int, best_val_cmax: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "rule_names": list(ALL_RULES),
            "metadata": {
                "size": args.size,
                "top_k": args.top_k,
                "run_id": args.run_id,
                "best_val_cmax": best_val_cmax,
                "type": "rule_selector_ppo",
            },
        },
        path,
    )


def train(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_fixed_dataset([args.size], ["train", "val"])
    train_instances = load_fixed_instances(args.size, "train")
    val_instances = load_fixed_instances(args.size, "val")
    if args.max_instances is not None:
        train_instances = train_instances[: max(1, args.max_instances)]
        val_instances = val_instances[: max(1, args.max_instances)]
    if args.dry_run:
        train_instances = train_instances[: max(1, args.max_instances or 1)]
        val_instances = val_instances[:1]

    probe_env = RuleSelectorEnv(
        train_instances[0],
        top_k=args.top_k,
        mlp_soft_model_path=args.mlp_soft_model_path,
        mlp_pairwise_model_path=args.mlp_pairwise_model_path,
    )
    state_dim = len(probe_env.reset(train_instances[0]))
    action_dim = probe_env.action_dim
    model = RuleActorCritic(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    resolved_run_id = make_run_id(args.run_id)
    args.run_id = resolved_run_id
    ckpt_dir = make_run_dir(
        Path("checkpoints/stage_F/rule_selector_ppo"),
        [args.size, f"topk{args.top_k}"],
        f"runid{resolved_run_id}",
        overwrite=args.overwrite,
    )
    latest_dir = Path("checkpoints/stage_F/rule_selector_ppo") / f"{args.size}_topk{args.top_k}_latest"
    log_dir = Path("logs/stage_F/rule_selector_ppo")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = make_result_path(log_dir, "train", [args.size, f"topk{args.top_k}", f"runid{resolved_run_id}"], None, overwrite=args.overwrite)

    fieldnames = [
        "episode",
        "instance_id",
        "episode_reward",
        "final_cmax",
        "fifo_cmax",
        "val_cmax",
        "policy_loss",
        "value_loss",
        "entropy",
        "grad_norm",
    ]
    best_val = float("inf")
    buffer = RuleRolloutBuffer()
    rows = []
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        iterator = progress_iter(range(1, args.episodes + 1), desc=f"train-rule-selector {args.size}", total=args.episodes)
        for episode in iterator:
            instance = train_instances[(episode - 1) % len(train_instances)]
            env = RuleSelectorEnv(
                instance,
                top_k=args.top_k,
                mlp_soft_model_path=args.mlp_soft_model_path,
                mlp_pairwise_model_path=args.mlp_pairwise_model_path,
                seed=args.seed + episode,
            )
            state = env.reset(instance)
            episode_reward = 0.0
            while not env.env.is_done():
                mask = env.action_mask()
                action, logprob, value = select_action(model, state, mask, device, greedy=False)
                next_state, reward, done, _ = env.step(action)
                buffer.add(state, action, mask, logprob, reward, done, value)
                state = next_state
                episode_reward += reward
            stats = update_model(
                model,
                optimizer,
                buffer,
                args.gamma,
                args.gae_lambda,
                args.clip_ratio,
                args.entropy_coef,
                args.value_coef,
                args.update_epochs,
                args.batch_size,
                device,
            )
            val_cmax = evaluate_validation(model, val_instances, args, device, max_eval_instances=1 if args.dry_run else 5)
            if val_cmax < best_val:
                best_val = val_cmax
                save_checkpoint(ckpt_dir / "best.pt", model, args, state_dim, action_dim, best_val)
            row = {
                "episode": episode,
                "instance_id": getattr(instance, "instance_id", getattr(instance, "name", "")),
                "episode_reward": episode_reward,
                "final_cmax": float(env.env.current_cmax),
                "fifo_cmax": float(env.fifo_cmax),
                "val_cmax": val_cmax,
                **stats,
            }
            writer.writerow(row)
            rows.append(row)
            if args.dry_run and episode >= args.episodes:
                break

    save_checkpoint(ckpt_dir / "last.pt", model, args, state_dim, action_dim, best_val)
    update_latest_dir(ckpt_dir, latest_dir)
    summary = [{
        "size": args.size,
        "top_k": args.top_k,
        "run_id": resolved_run_id,
        "episodes": len(rows),
        "best_val_cmax": best_val,
        "final_train_cmax": rows[-1]["final_cmax"] if rows else 0.0,
    }]
    summary_fields = ["size", "top_k", "run_id", "episodes", "best_val_cmax", "final_train_cmax"]
    summary_dir = Path("logs/stage_F/rule_selector_ppo")
    write_csv(summary, summary_dir / "train_summary_latest.csv", summary_fields)
    all_path = summary_dir / "train_summary_all.csv"
    existing = []
    if all_path.exists():
        with all_path.open("r", newline="") as f:
            existing = list(csv.DictReader(f))
    write_csv([*existing, *summary], all_path, summary_fields)
    print(f"Saved checkpoints to {ckpt_dir}")
    print(f"Updated latest checkpoint dir: {latest_dir}")
    print(f"Saved training log to {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--mlp_soft_model_path", default=None)
    parser.add_argument("--mlp_pairwise_model_path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
