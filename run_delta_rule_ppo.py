"""Standalone Delta-Rule PPO runner for HGCR-PPO Stage F2-2."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from instance_manager import SIZES, SPLITS, ensure_fixed_dataset, load_fixed_instances
from rule_selector_env import RuleSelectorEnv
from schedule_validator import validate_schedule
from src.evaluation.metrics import compute_metrics
from utils.experiment_io import progress_iter, write_csv


RESULT_FIELDS = [
    "instance_id",
    "seed",
    "size",
    "split",
    "top_k",
    "method",
    "final_cmax",
    "ranker_baseline_cmax",
    "delta_improvement",
    "action_keep_ranker_ratio",
    "action_fifo_ratio",
    "action_lookahead_ratio",
    "action_greedy_ect_ratio",
    "action_minload_ratio",
    "override_ratio",
    "fallback_count",
]
TRAIN_FIELDS = [
    "episode",
    "seed",
    "size",
    "split",
    "top_k",
    "final_cmax",
    "ranker_baseline_cmax",
    "delta_improvement",
    "total_reward",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_frac",
    "keep_ranker_ratio",
    "override_ratio",
    "fallback_count",
]
ACTION_FIELDS = [
    "seed",
    "size",
    "split",
    "top_k",
    "total_decisions",
    "keep_ranker_count",
    "fifo_override_count",
    "lookahead_override_count",
    "greedy_ect_override_count",
    "minload_override_count",
    "override_count",
    "fallback_count",
    "keep_ranker_ratio",
    "override_ratio",
]


@dataclass
class DeltaBuffer:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def add(self, state, action, mask, logprob, reward, value, done) -> None:
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(int(action))
        self.masks.append(np.asarray(mask, dtype=bool).copy())
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.masks.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def tensors(self, gamma: float, gae_lambda: float, device: torch.device) -> Dict[str, torch.Tensor]:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray([*self.values, 0.0], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for idx in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[idx]
            delta = rewards[idx] + gamma * values[idx + 1] * nonterminal - values[idx]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[idx] = gae
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


class DeltaRulePPOAgent(nn.Module):
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

    def dist_value(self, states: torch.Tensor, masks: torch.Tensor):
        features = self.shared(states)
        logits = self.actor(features).masked_fill(~masks.bool(), -1e9)
        values = self.critic(features).squeeze(-1)
        return torch.distributions.Categorical(logits=logits), values


def select_action(model: DeltaRulePPOAgent, state: np.ndarray, mask: np.ndarray, device: torch.device, greedy: bool):
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
    with torch.no_grad():
        dist, value = model.dist_value(state_t, mask_t)
        action = torch.argmax(dist.probs, dim=-1) if greedy else dist.sample()
        prob = dist.probs[0, int(action.item())]
    return int(action.item()), float(dist.log_prob(action).item()), float(value.item()), float(prob.item())


def update_model(model, optimizer, buffer: DeltaBuffer, args, device: torch.device) -> Dict[str, float]:
    if not buffer.rewards:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    batch = buffer.tensors(args.gamma, args.gae_lambda, device)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    updates = 0
    n = len(batch["actions"])
    for _ in range(args.update_epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, min(args.batch_size, n)):
            idx = order[start : start + min(args.batch_size, n)]
            dist, values = model.dist_value(batch["states"][idx], batch["masks"][idx])
            logprobs = dist.log_prob(batch["actions"][idx])
            ratio = torch.exp(logprobs - batch["old_logprobs"][idx])
            unclipped = ratio * batch["advantages"][idx]
            clipped = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * batch["advantages"][idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.smooth_l1_loss(values, batch["returns"][idx])
            entropy = dist.entropy().mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (batch["old_logprobs"][idx] - logprobs).mean()
                clip_frac = ((ratio < 1.0 - args.clip_ratio) | (ratio > 1.0 + args.clip_ratio)).float().mean()
            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy.item())
            stats["approx_kl"] += float(approx_kl.item())
            stats["clip_frac"] += float(clip_frac.item())
            updates += 1
    buffer.clear()
    return {key: value / max(1, updates) for key, value in stats.items()}


def make_env(instance, args, seed: int) -> RuleSelectorEnv:
    return RuleSelectorEnv(
        instance,
        top_k=args.top_k,
        mlp_soft_model_path=args.ranker_ckpt,
        action_mode="delta_rule",
        baseline_type=args.baseline_type,
        include_pairwise_ranker=args.include_pairwise_ranker,
        seed=seed,
    )


def run_episode(model, instance, args, device: torch.device, seed: int, train_mode: bool):
    env = make_env(instance, args, seed)
    state = env.reset(instance)
    buffer = DeltaBuffer()
    total_reward = 0.0
    while not env.env.is_done():
        mask = env.action_mask()
        action, logprob, value, prob = select_action(model, state, mask, device, greedy=not train_mode)
        next_state, _, done, _ = env.step(action, action_probability=prob)
        buffer.add(state, action, mask, logprob, 0.0, value, done)
        state = next_state
    final_cmax = float(env.env.current_cmax)
    ranker_cmax = float(env.ranker_cmax)
    final_reward = ranker_cmax - final_cmax if args.reward_mode == "final_delta" else ranker_cmax - final_cmax
    if buffer.rewards:
        buffer.rewards[-1] = final_reward
    total_reward += final_reward
    diagnostics = env.diagnostics()
    return env, buffer, total_reward, diagnostics


def action_stats_row(args, diagnostics: Dict, seed: int) -> Dict:
    counts = diagnostics["executed_rule_counts"]
    raw = diagnostics["executed_rule_distribution"]
    total_decisions = int(diagnostics["total_decisions"])
    keep = counts.get("keep_ranker", 0.0)
    override_count = (
        counts.get("switch_to_fifo", 0.0)
        + counts.get("switch_to_lookahead", 0.0)
        + counts.get("switch_to_greedy_ect", 0.0)
        + counts.get("switch_to_minload", 0.0)
    )
    override = diagnostics["effective_switch_ratio"]
    return {
        "seed": seed,
        "size": args.size,
        "split": args.split,
        "top_k": args.top_k,
        "total_decisions": total_decisions,
        "keep_ranker_count": keep,
        "fifo_override_count": counts.get("switch_to_fifo", 0.0),
        "lookahead_override_count": counts.get("switch_to_lookahead", 0.0),
        "greedy_ect_override_count": counts.get("switch_to_greedy_ect", 0.0),
        "minload_override_count": counts.get("switch_to_minload", 0.0),
        "override_count": override_count,
        "fallback_count": diagnostics["fallback_count"],
        "keep_ranker_ratio": keep,
        "override_ratio": override,
    }


def train_and_eval(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_fixed_dataset([args.size], [args.split])
    instances = load_fixed_instances(args.size, args.split)
    if args.smoke_test:
        instances = instances[:1]

    probe = make_env(instances[0], args, args.seed)
    state_dim = len(probe.reset(instances[0]))
    action_dim = probe.action_dim
    model = DeltaRulePPOAgent(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_rows = []
    action_rows = []
    for episode in progress_iter(range(1, args.episodes + 1), desc="delta-rule-ppo train", total=args.episodes):
        instance = instances[(episode - 1) % len(instances)]
        env, buffer, total_reward, diagnostics = run_episode(model, instance, args, device, args.seed + episode, train_mode=True)
        stats = update_model(model, optimizer, buffer, args, device)
        final_cmax = float(env.env.current_cmax)
        ranker_cmax = float(env.ranker_cmax)
        train_rows.append(
            {
                "episode": episode,
                "seed": args.seed,
                "size": args.size,
                "split": args.split,
                "top_k": args.top_k,
                "final_cmax": final_cmax,
                "ranker_baseline_cmax": ranker_cmax,
                "delta_improvement": ranker_cmax - final_cmax,
                "total_reward": total_reward,
                "keep_ranker_ratio": diagnostics["keep_ranker_ratio"],
                "override_ratio": diagnostics["effective_switch_ratio"],
                "fallback_count": diagnostics["fallback_count"],
                **stats,
            }
        )
        action_rows.append(action_stats_row(args, diagnostics, args.seed))

    eval_rows = []
    for instance in progress_iter(instances, desc="delta-rule-ppo eval", total=len(instances)):
        env, _, _, diagnostics = run_episode(model, instance, args, device, args.seed, train_mode=False)
        metrics = compute_metrics(env.env)
        validation = validate_schedule(env.env, instance)
        final_cmax = float(metrics["Cmax_roll"])
        ranker_cmax = float(env.ranker_cmax)
        eval_rows.append(
            {
                "instance_id": getattr(instance, "instance_id", getattr(instance, "name", "")),
                "seed": args.seed,
                "size": args.size,
                "split": args.split,
                "top_k": args.top_k,
                "method": "DeltaRulePPO",
                "final_cmax": final_cmax,
                "ranker_baseline_cmax": ranker_cmax,
                "delta_improvement": ranker_cmax - final_cmax,
                "action_keep_ranker_ratio": diagnostics["keep_ranker_ratio"],
                "action_fifo_ratio": diagnostics["switch_to_fifo_ratio"],
                "action_lookahead_ratio": diagnostics["switch_to_lookahead_ratio"],
                "action_greedy_ect_ratio": diagnostics["switch_to_greedy_ratio"],
                "action_minload_ratio": diagnostics["switch_to_minload_ratio"],
                "override_ratio": diagnostics["effective_switch_ratio"],
                "fallback_count": diagnostics["fallback_count"],
                "is_valid_schedule": validation["is_valid_schedule"],
                "diagnostics": json.dumps(diagnostics, sort_keys=True),
            }
        )

    write_csv(train_rows, output_dir / "delta_rule_ppo_train_log.csv", TRAIN_FIELDS)
    write_csv(eval_rows, output_dir / "delta_rule_ppo_eval.csv", [*RESULT_FIELDS, "is_valid_schedule", "diagnostics"])
    write_csv(summarize_eval(eval_rows), output_dir / "delta_rule_ppo_eval_summary.csv", summary_fields())
    write_csv(action_rows, output_dir / "delta_rule_action_stats.csv", ACTION_FIELDS)
    print(f"Saved Delta-Rule PPO outputs to {output_dir}")


def summarize_eval(rows: Iterable[Dict]) -> List[Dict]:
    rows = list(rows)
    if not rows:
        return []
    out = {
        "method": "DeltaRulePPO",
        "size": rows[0]["size"],
        "split": rows[0]["split"],
        "top_k": rows[0]["top_k"],
    }
    metrics = {
        "Cmax": [float(row["final_cmax"]) for row in rows],
        "ranker_baseline": [float(row["ranker_baseline_cmax"]) for row in rows],
        "delta_improvement": [float(row["delta_improvement"]) for row in rows],
        "keep_ranker_ratio": [float(row["action_keep_ranker_ratio"]) for row in rows],
        "override_ratio": [float(row["override_ratio"]) for row in rows],
        "fallback_count": [float(row["fallback_count"]) for row in rows],
    }
    out["Cmax_mean"] = mean(metrics["Cmax"])
    out["Cmax_std"] = pstdev(metrics["Cmax"]) if len(rows) > 1 else 0.0
    out["ranker_baseline_mean"] = mean(metrics["ranker_baseline"])
    out["delta_improvement_mean"] = mean(metrics["delta_improvement"])
    out["delta_improvement_std"] = pstdev(metrics["delta_improvement"]) if len(rows) > 1 else 0.0
    out["keep_ranker_ratio_mean"] = mean(metrics["keep_ranker_ratio"])
    out["override_ratio_mean"] = mean(metrics["override_ratio"])
    out["fallback_count_mean"] = mean(metrics["fallback_count"])
    return [out]


def summary_fields() -> List[str]:
    return [
        "method",
        "size",
        "split",
        "top_k",
        "Cmax_mean",
        "Cmax_std",
        "ranker_baseline_mean",
        "delta_improvement_mean",
        "delta_improvement_std",
        "keep_ranker_ratio_mean",
        "override_ratio_mean",
        "fallback_count_mean",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, default="small")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ranker_ckpt", default="checkpoints/stage_C/mlp_ranker/small_topk5_soft_ce/best.pt")
    parser.add_argument("--baseline_type", choices=["ranker"], default="ranker")
    parser.add_argument("--reward_mode", choices=["final_delta", "step_delta"], default="final_delta")
    parser.add_argument("--output_dir", default="data/results/stage_F/delta_rule_ppo")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--include_pairwise_ranker", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    args = parser.parse_args()
    train_and_eval(args)


if __name__ == "__main__":
    main()
