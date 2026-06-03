"""Standalone Conservative Delta-Rule PPO runner for HGCR-PPO Stage F2-2.1."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
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


METHOD = "DeltaRulePPO"
ACTION_NAMES = ["keep_ranker", "switch_to_fifo", "switch_to_lookahead", "switch_to_greedy_ect", "switch_to_minload"]
TRAIN_FIELDS = [
    "run_id",
    "episode",
    "seed",
    "size",
    "split",
    "top_k",
    "final_cmax",
    "ranker_baseline_cmax",
    "delta_improvement",
    "normalized_delta_improvement",
    "override_penalty",
    "penalty_value",
    "reward_mode",
    "total_reward",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_frac",
    "keep_ranker_ratio",
    "override_ratio",
    "fallback_count",
    "checkpoint_type",
    "best_so_far",
    "eval_delta_improvement_mean",
    "eval_Cmax_mean",
    "is_best_checkpoint",
]
EVAL_FIELDS = [
    "run_id",
    "checkpoint_type",
    "instance_id",
    "seed",
    "size",
    "split",
    "top_k",
    "method",
    "final_cmax",
    "ranker_baseline_cmax",
    "delta_improvement",
    "normalized_delta_improvement",
    "action_keep_ranker_ratio",
    "action_fifo_ratio",
    "action_lookahead_ratio",
    "action_greedy_ect_ratio",
    "action_minload_ratio",
    "override_ratio",
    "fallback_count",
    "is_valid_schedule",
]
SUMMARY_FIELDS = [
    "run_id",
    "method",
    "checkpoint_type",
    "size",
    "split",
    "top_k",
    "episodes",
    "seed",
    "Cmax_mean",
    "Cmax_std",
    "ranker_baseline_mean",
    "delta_improvement_mean",
    "delta_improvement_std",
    "normalized_delta_improvement_mean",
    "keep_ranker_ratio_mean",
    "override_ratio_mean",
    "fallback_count_mean",
    "valid_schedule_rate",
]
ACTION_FIELDS = [
    "run_id",
    "episode",
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
    "fifo_override_ratio",
    "lookahead_override_ratio",
    "greedy_ect_override_ratio",
    "minload_override_ratio",
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


def initialize_conservative_actor_bias(agent: DeltaRulePPOAgent, keep_ranker_bias: float) -> None:
    with torch.no_grad():
        agent.actor.bias.zero_()
        if agent.actor.bias.numel() > 0:
            agent.actor.bias[0] = float(keep_ranker_bias)


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


def make_run_id(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    reward_token = args.reward_mode.replace("_", "")
    return (
        f"{METHOD}_{args.size}_{args.split}_topk{args.top_k}_ep{args.episodes}_seed{args.seed}_"
        f"{args.baseline_type}_{reward_token}_{timestamp}_{suffix}"
    )


def prepare_run_dir(args) -> tuple[str, Path]:
    base_dir = Path(args.output_dir) / "runs"
    base_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        run_id = make_run_id(args)
        run_dir = base_dir / run_id
        if not run_dir.exists():
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_id, run_dir
    raise RuntimeError("Could not create a unique DeltaRulePPO run directory after several attempts.")


def make_output_paths(run_dir: Path, run_id: str) -> Dict[str, Path]:
    return {
        "train_log": run_dir / f"train_log__{run_id}.csv",
        "eval_last": run_dir / f"eval_last__{run_id}.csv",
        "eval_best": run_dir / f"eval_best__{run_id}.csv",
        "eval_summary_last": run_dir / f"eval_summary_last__{run_id}.csv",
        "eval_summary_best": run_dir / f"eval_summary_best__{run_id}.csv",
        "action_stats": run_dir / f"action_stats__{run_id}.csv",
        "manifest": run_dir / f"manifest__{run_id}.json",
        "best_checkpoint": Path("checkpoints/stage_F/delta_rule_ppo") / run_id / "best.pt",
        "last_checkpoint": Path("checkpoints/stage_F/delta_rule_ppo") / run_id / "last.pt",
    }


def write_no_overwrite(rows: Iterable[Dict], path: Path, fieldnames: List[str]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {path}")
    write_csv(rows, path, fieldnames)


def make_env(instance, args, seed: int) -> RuleSelectorEnv:
    return RuleSelectorEnv(
        instance,
        top_k=args.top_k,
        mlp_soft_model_path=args.ranker_ckpt,
        action_mode="delta_rule",
        baseline_type=args.baseline_type,
        include_pairwise_ranker=False,
        seed=seed,
    )


def reward_values(ranker_cmax: float, final_cmax: float, override_ratio: float, args) -> Dict[str, float]:
    delta = ranker_cmax - final_cmax
    normalized_delta = delta / max(ranker_cmax, 1e-8) * 100.0
    penalty_value = float(args.override_penalty) * float(override_ratio)
    if args.reward_mode == "conservative_final_delta":
        total_reward = normalized_delta - penalty_value
    else:
        total_reward = delta
        penalty_value = 0.0
    return {
        "delta_improvement": delta,
        "normalized_delta_improvement": normalized_delta,
        "penalty_value": penalty_value,
        "total_reward": total_reward,
    }


def run_episode(model, instance, args, device: torch.device, seed: int, train_mode: bool):
    env = make_env(instance, args, seed)
    state = env.reset(instance)
    buffer = DeltaBuffer()
    while not env.env.is_done():
        mask = env.action_mask()
        action, logprob, value, prob = select_action(model, state, mask, device, greedy=not train_mode)
        next_state, _, done, _ = env.step(action, action_probability=prob)
        buffer.add(state, action, mask, logprob, 0.0, value, done)
        state = next_state
    final_cmax = float(env.env.current_cmax)
    ranker_cmax = float(env.ranker_cmax)
    diagnostics = env.diagnostics()
    reward_info = reward_values(ranker_cmax, final_cmax, float(diagnostics["effective_switch_ratio"]), args)
    if buffer.rewards:
        buffer.rewards[-1] = float(reward_info["total_reward"])
    return env, buffer, diagnostics, reward_info


def action_stats_row(args, diagnostics: Dict, episode: int, seed: int, run_id: str) -> Dict:
    counts = diagnostics["executed_rule_counts"]
    total = max(1.0, float(diagnostics["total_decisions"]))
    keep = float(counts.get("keep_ranker", 0.0))
    fifo = float(counts.get("switch_to_fifo", 0.0))
    lookahead = float(counts.get("switch_to_lookahead", 0.0))
    greedy = float(counts.get("switch_to_greedy_ect", 0.0))
    minload = float(counts.get("switch_to_minload", 0.0))
    override_count = fifo + lookahead + greedy + minload
    return {
        "run_id": run_id,
        "episode": episode,
        "seed": seed,
        "size": args.size,
        "split": args.split,
        "top_k": args.top_k,
        "total_decisions": int(diagnostics["total_decisions"]),
        "keep_ranker_count": keep,
        "fifo_override_count": fifo,
        "lookahead_override_count": lookahead,
        "greedy_ect_override_count": greedy,
        "minload_override_count": minload,
        "override_count": override_count,
        "fallback_count": float(diagnostics["fallback_count"]),
        "keep_ranker_ratio": keep / total,
        "fifo_override_ratio": fifo / total,
        "lookahead_override_ratio": lookahead / total,
        "greedy_ect_override_ratio": greedy / total,
        "minload_override_ratio": minload / total,
        "override_ratio": override_count / total,
    }


def save_checkpoint(path: Path, model, args, run_id: str, state_dim: int, action_dim: int, checkpoint_type: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_names": ACTION_NAMES,
            "metadata": {
                "run_id": run_id,
                "checkpoint_type": checkpoint_type,
                "method": METHOD,
                "size": args.size,
                "split": args.split,
                "top_k": args.top_k,
                "episodes": args.episodes,
                "seed": args.seed,
                "baseline_type": args.baseline_type,
                "reward_mode": args.reward_mode,
                "keep_ranker_bias": args.keep_ranker_bias,
                "override_penalty": args.override_penalty,
            },
        },
        path,
    )


def load_checkpoint(path: Path, state_dim: int, action_dim: int, device: torch.device) -> DeltaRulePPOAgent:
    model = DeltaRulePPOAgent(state_dim, action_dim).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_policy(model, instances, args, device: torch.device, run_id: str, checkpoint_type: str) -> List[Dict]:
    rows = []
    for instance in instances:
        env, _, diagnostics, reward_info = run_episode(model, instance, args, device, args.seed, train_mode=False)
        metrics = compute_metrics(env.env)
        validation = validate_schedule(env.env, instance)
        rows.append(
            {
                "run_id": run_id,
                "checkpoint_type": checkpoint_type,
                "instance_id": getattr(instance, "instance_id", getattr(instance, "name", "")),
                "seed": args.seed,
                "size": args.size,
                "split": args.split,
                "top_k": args.top_k,
                "method": METHOD,
                "final_cmax": float(metrics["Cmax_roll"]),
                "ranker_baseline_cmax": float(env.ranker_cmax),
                "delta_improvement": float(reward_info["delta_improvement"]),
                "normalized_delta_improvement": float(reward_info["normalized_delta_improvement"]),
                "action_keep_ranker_ratio": float(diagnostics["keep_ranker_ratio"]),
                "action_fifo_ratio": float(diagnostics["switch_to_fifo_ratio"]),
                "action_lookahead_ratio": float(diagnostics["switch_to_lookahead_ratio"]),
                "action_greedy_ect_ratio": float(diagnostics["switch_to_greedy_ratio"]),
                "action_minload_ratio": float(diagnostics["switch_to_minload_ratio"]),
                "override_ratio": float(diagnostics["effective_switch_ratio"]),
                "fallback_count": float(diagnostics["fallback_count"]),
                "is_valid_schedule": bool(validation["is_valid_schedule"]),
            }
        )
    return rows


def summarize_eval(rows: Iterable[Dict], args, run_id: str, checkpoint_type: str) -> List[Dict]:
    rows = list(rows)
    if not rows:
        return []
    valid = [1.0 if row["is_valid_schedule"] else 0.0 for row in rows]
    values = {
        "Cmax": [float(row["final_cmax"]) for row in rows],
        "ranker": [float(row["ranker_baseline_cmax"]) for row in rows],
        "delta": [float(row["delta_improvement"]) for row in rows],
        "normalized_delta": [float(row["normalized_delta_improvement"]) for row in rows],
        "keep": [float(row["action_keep_ranker_ratio"]) for row in rows],
        "override": [float(row["override_ratio"]) for row in rows],
        "fallback": [float(row["fallback_count"]) for row in rows],
    }
    return [
        {
            "run_id": run_id,
            "method": METHOD,
            "checkpoint_type": checkpoint_type,
            "size": args.size,
            "split": args.split,
            "top_k": args.top_k,
            "episodes": args.episodes,
            "seed": args.seed,
            "Cmax_mean": mean(values["Cmax"]),
            "Cmax_std": pstdev(values["Cmax"]) if len(rows) > 1 else 0.0,
            "ranker_baseline_mean": mean(values["ranker"]),
            "delta_improvement_mean": mean(values["delta"]),
            "delta_improvement_std": pstdev(values["delta"]) if len(rows) > 1 else 0.0,
            "normalized_delta_improvement_mean": mean(values["normalized_delta"]),
            "keep_ranker_ratio_mean": mean(values["keep"]),
            "override_ratio_mean": mean(values["override"]),
            "fallback_count_mean": mean(values["fallback"]),
            "valid_schedule_rate": mean(valid),
        }
    ]


def get_git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return ""


def write_manifest(path: Path, args, run_id: str, start_time: str, end_time: str, output_paths: Dict[str, Path], device: torch.device) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {path}")
    manifest = {
        "run_id": run_id,
        "start_time": start_time,
        "end_time": end_time,
        "command_args": vars(args),
        "method": METHOD,
        "size": args.size,
        "split": args.split,
        "top_k": args.top_k,
        "episodes": args.episodes,
        "seed": args.seed,
        "baseline_type": args.baseline_type,
        "reward_mode": args.reward_mode,
        "output_files": {key: str(value) for key, value in output_paths.items()},
        "git_commit_hash": get_git_commit(),
        "python_version": sys.version,
        "device": str(device),
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def train_and_eval(args) -> None:
    start_time = datetime.now().isoformat(timespec="seconds")
    run_id, run_dir = prepare_run_dir(args)
    output_paths = make_output_paths(run_dir, run_id)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    ensure_fixed_dataset([args.size], [args.split])
    instances = load_fixed_instances(args.size, args.split)
    if args.smoke_test:
        instances = instances[:1]

    probe = make_env(instances[0], args, args.seed)
    state_dim = len(probe.reset(instances[0]))
    action_dim = probe.action_dim
    model = DeltaRulePPOAgent(state_dim, action_dim).to(device)
    initialize_conservative_actor_bias(model, args.keep_ranker_bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_rows = []
    action_rows = []
    best_score = -float("inf")
    best_eval_rows = []
    best_state_dict = None
    patience_counter = 0

    for episode in progress_iter(range(1, args.episodes + 1), desc="conservative-delta-rule train", total=args.episodes):
        instance = instances[(episode - 1) % len(instances)]
        env, buffer, diagnostics, reward_info = run_episode(model, instance, args, device, args.seed + episode, train_mode=True)
        stats = update_model(model, optimizer, buffer, args, device)

        eval_delta_mean = ""
        eval_cmax_mean = ""
        is_best = False
        if episode % max(1, args.eval_interval) == 0 or episode == args.episodes:
            eval_rows = evaluate_policy(model, instances, args, device, run_id, "interval")
            eval_summary = summarize_eval(eval_rows, args, run_id, "interval")
            eval_delta_mean = eval_summary[0]["delta_improvement_mean"] if eval_summary else ""
            eval_cmax_mean = eval_summary[0]["Cmax_mean"] if eval_summary else ""
            if eval_summary and float(eval_delta_mean) > best_score:
                best_score = float(eval_delta_mean)
                best_eval_rows = eval_rows
                best_state_dict = deepcopy(model.state_dict())
                is_best = True
                patience_counter = 0
            else:
                patience_counter += 1
            if args.early_stop and patience_counter >= args.early_stop_patience:
                print(f"Early stop at episode {episode}: no eval improvement for {patience_counter} checks.")
                args.episodes = episode
                break

        final_cmax = float(env.env.current_cmax)
        ranker_cmax = float(env.ranker_cmax)
        train_rows.append(
            {
                "run_id": run_id,
                "episode": episode,
                "seed": args.seed,
                "size": args.size,
                "split": args.split,
                "top_k": args.top_k,
                "final_cmax": final_cmax,
                "ranker_baseline_cmax": ranker_cmax,
                "delta_improvement": reward_info["delta_improvement"],
                "normalized_delta_improvement": reward_info["normalized_delta_improvement"],
                "override_penalty": args.override_penalty,
                "penalty_value": reward_info["penalty_value"],
                "reward_mode": args.reward_mode,
                "total_reward": reward_info["total_reward"],
                "keep_ranker_ratio": diagnostics["keep_ranker_ratio"],
                "override_ratio": diagnostics["effective_switch_ratio"],
                "fallback_count": diagnostics["fallback_count"],
                "checkpoint_type": "best" if is_best else "",
                "best_so_far": best_score if best_score > -float("inf") else "",
                "eval_delta_improvement_mean": eval_delta_mean,
                "eval_Cmax_mean": eval_cmax_mean,
                "is_best_checkpoint": is_best,
                **stats,
            }
        )
        action_rows.append(action_stats_row(args, diagnostics, episode, args.seed, run_id))

    last_state_dict = deepcopy(model.state_dict())
    save_checkpoint(output_paths["last_checkpoint"], model, args, run_id, state_dim, action_dim, "last")
    last_rows = evaluate_policy(model, instances, args, device, run_id, "last")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    else:
        model.load_state_dict(last_state_dict)
    save_checkpoint(output_paths["best_checkpoint"], model, args, run_id, state_dim, action_dim, "best")
    best_eval_rows = evaluate_policy(model, instances, args, device, run_id, "best")

    write_no_overwrite(train_rows, output_paths["train_log"], TRAIN_FIELDS)
    write_no_overwrite(last_rows, output_paths["eval_last"], EVAL_FIELDS)
    write_no_overwrite(best_eval_rows, output_paths["eval_best"], EVAL_FIELDS)
    write_no_overwrite(summarize_eval(last_rows, args, run_id, "last"), output_paths["eval_summary_last"], SUMMARY_FIELDS)
    write_no_overwrite(summarize_eval(best_eval_rows, args, run_id, "best"), output_paths["eval_summary_best"], SUMMARY_FIELDS)
    write_no_overwrite(action_rows, output_paths["action_stats"], ACTION_FIELDS)
    write_manifest(output_paths["manifest"], args, run_id, start_time, datetime.now().isoformat(timespec="seconds"), output_paths, device)
    print(f"Saved Conservative Delta-Rule PPO run to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, default="small")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ranker_ckpt", default="checkpoints/stage_C/mlp_ranker/small_topk5_soft_ce/best.pt")
    parser.add_argument("--baseline_type", choices=["ranker"], default="ranker")
    parser.add_argument("--reward_mode", choices=["conservative_final_delta", "final_delta", "step_delta"], default="conservative_final_delta")
    parser.add_argument("--override_penalty", type=float, default=0.05)
    parser.add_argument("--keep_ranker_bias", type=float, default=2.0)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--output_dir", default="data/results/stage_F/delta_rule_ppo")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke_test", action="store_true")
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
