"""Official dynamic HGCR-PPO training entry.

The action space follows recent dynamic scheduling DRL practice: PPO selects
one dispatching rule from a compact rule set, and the selected rule chooses the
job to schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import uuid
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from candidate_generator import generate_candidates
from dynamic_rolling_scenarios import LEVELS, generate_dynamic_scenarios
from instance_manager import SIZES
from mlp_models import load_checkpoint
from schedule_validator import validate_schedule
from src.baselines.heuristics import choose_split_num, estimated_completion_time, lookahead_score, mean_candidate_processing_time
from src.envs.rolling_scheduling_env import RollingSchedulingEnv
from src.evaluation.metrics import compute_metrics
from stage_c_utils import extract_candidate_features
from utils.experiment_io import progress_iter, write_csv


RULE_NAMES = ["FIFO", "GreedyECT", "Lookahead", "MLP_Ranker_soft_ce"]
REWARD_MODES = ["env_cmax_delta", "final_delta", "util_plus_cmax"]
TRAIN_FIELDS = [
    "episode",
    "episode_reward",
    "episode_Cmax",
    "machine_utilization",
    "step_reward_sum",
    "final_reward",
    "total_reward",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "is_eval_step",
]
EVAL_HISTORY_FIELDS = [
    "eval_step",
    "episode",
    "size",
    "seed",
    "arrival_intensity",
    "carryover_ratio",
    "reward_beta",
    "eval_Cmax_mean",
    "eval_Cmax_std",
    "eval_reward_mean",
    "eval_reward_std",
    "best_so_far_Cmax",
    "baseline_FIFO_Cmax",
    "baseline_MLPRanker_Cmax",
]
EVAL_FIELDS = [
    "method",
    "scenario_type",
    "size",
    "arrival_intensity",
    "carryover_ratio",
    "Cmax_mean",
    "Cmax_std",
    "relative_to_FIFO",
    "relative_to_MLPRanker",
    "valid_schedule_rate",
    "runtime_mean",
]
ACTION_FIELDS = ["rule_name", "selection_count", "selection_ratio"]
CURVE_FIELDS = ["episode", "episode_reward", "episode_Cmax", "step_reward_sum", "final_reward", "total_reward"]
ACTION_HISTORY_FIELDS = ["episode", "decision_index", "size", "seed", "arrival_intensity", "carryover_ratio", "reward_beta", "action_id", "action_name", "is_eval"]
ACTION_STAGE_FIELDS = ["stage_start_episode", "stage_end_episode", "size", "seed", "arrival_intensity", "carryover_ratio", "reward_beta", "action_name", "action_count", "action_ratio"]
ACTION_DISPLAY_NAMES = {"MLP_Ranker_soft_ce": "MLP-Ranker"}


@dataclass
class PPOBuffer:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def add(self, state, action: int, logprob: float, reward: float, value: float, done: bool) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def __len__(self) -> int:
        return len(self.actions)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def tensors(self, gamma: float, gae_lambda: float, device) -> Dict[str, torch.Tensor]:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray([*self.values, 0.0], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for idx in reversed(range(len(rewards))):
            delta = rewards[idx] + gamma * values[idx + 1] * (1.0 - dones[idx]) - values[idx]
            gae = delta + gamma * gae_lambda * (1.0 - dones[idx]) * gae
            advantages[idx] = gae
        returns = advantages + values[:-1]
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return {
            "states": torch.tensor(np.asarray(self.states), dtype=torch.float32, device=device),
            "actions": torch.tensor(self.actions, dtype=torch.long, device=device),
            "old_logprobs": torch.tensor(self.logprobs, dtype=torch.float32, device=device),
            "returns": torch.tensor(returns, dtype=torch.float32, device=device),
            "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
        }


class RuleActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, action_dim))
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def dist_value(self, states):
        logits = self.actor(states)
        value = self.critic(states).squeeze(-1)
        return torch.distributions.Categorical(logits=logits), value


def make_run_id(args) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"HGCRDYN_stageG_{args.size}_k{args.top_k}_e{args.episodes}_s{args.seed}"
        f"_{args.arrival_intensity}_{args.carryover_ratio}_b{args.reward_beta}_{stamp}_{uuid.uuid4().hex[:8]}"
    )


def output_paths(args, run_id: str) -> Dict[str, Path]:
    run_dir = Path(args.output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    # Keep the full metadata-rich run_id in the directory name and manifest.
    # Repeating it in every filename can exceed Windows' traditional MAX_PATH
    # limit on user workstations with deep project paths.
    preferred_eval_history = run_dir / f"eval_history__{run_id}.csv"
    eval_history_path = preferred_eval_history
    if len(str(preferred_eval_history.resolve())) > 240:
        eval_history_path = run_dir / "eval_history.csv"
    preferred_action_history = run_dir / f"action_history__{run_id}.csv"
    action_history_path = preferred_action_history if len(str(preferred_action_history.resolve())) <= 240 else run_dir / "action_history.csv"
    preferred_action_stage = run_dir / f"action_stage_summary__{run_id}.csv"
    action_stage_path = preferred_action_stage if len(str(preferred_action_stage.resolve())) <= 240 else run_dir / "action_stage_summary.csv"
    return {
        "run_dir": run_dir,
        "train": run_dir / "train_log.csv",
        "eval_summary": run_dir / "eval_summary.csv",
        "action_ratio": run_dir / "action_ratio.csv",
        "curve": run_dir / "reward_cmax_curve.csv",
        "eval_history": eval_history_path,
        "action_history": action_history_path,
        "action_stage": action_stage_path,
        "manifest": run_dir / "manifest.json",
        "checkpoint": run_dir / "hgcr_dynamic_ppo.pt",
    }


def load_ranker_or_none(path: str, device: str):
    ckpt = Path(path)
    if not ckpt.exists():
        return None
    return load_checkpoint(ckpt, device=device)


def reset_env_for_scenario(scenario: dict) -> RollingSchedulingEnv:
    env = RollingSchedulingEnv(scenario["instance"])
    env.reset(scenario["instance"])
    env.machine_available_time.update({k: float(v) for k, v in scenario["machine_initial_available_time"].items()})
    env.current_cmax = max(env.machine_available_time.values(), default=0.0)
    return env


def current_utilization(env: RollingSchedulingEnv) -> float:
    return float(compute_metrics(env)["machine_utilization"])


def _first_by_rule(env: RollingSchedulingEnv, rule_name: str, ranker_model=None, device: str = "cpu", top_k: int = 5) -> str:
    jobs = env.get_schedulable_jobs()
    if not jobs:
        raise RuntimeError("No schedulable jobs available.")
    if rule_name == "FIFO":
        return min(jobs, key=lambda j: (env.job_by_id[j].release_time, j))
    if rule_name == "GreedyECT":
        return min(jobs, key=lambda j: (estimated_completion_time(env, j), j))
    if rule_name == "Lookahead":
        return min(jobs, key=lambda j: (lookahead_score(env, j), j))
    if rule_name == "MLP_Ranker_soft_ce" and ranker_model is not None:
        candidates = generate_candidates(env, candidate_mode="hybrid_topk", top_k=top_k, fallback_to_all=True)
        features = torch.tensor([extract_candidate_features(env, candidates)], dtype=torch.float32, device=device)
        with torch.no_grad():
            scores = ranker_model(features).squeeze(0).detach().cpu().numpy()
        best_idx = int(np.argmin(scores))
        return candidates[best_idx]
    return min(jobs, key=lambda j: (estimated_completion_time(env, j), j))


def rule_choices(env: RollingSchedulingEnv, ranker_model=None, device: str = "cpu", top_k: int = 5) -> List[str]:
    return [_first_by_rule(env, rule, ranker_model=ranker_model, device=device, top_k=top_k) for rule in RULE_NAMES]


def state_features(env: RollingSchedulingEnv, choices: Sequence[str], ranker_available: bool) -> np.ndarray:
    schedulable = env.get_schedulable_jobs()
    total_jobs = max(1, len(env.instance.jobs))
    loads = list(env.machine_available_time.values())
    now = float(env._current_decision_time())
    max_release = max((job.release_time for job in env.instance.jobs), default=1.0) or 1.0
    base = [
        len(env.scheduled_jobs) / total_jobs,
        len(env.unscheduled_jobs) / total_jobs,
        len(schedulable) / total_jobs,
        float(env.current_cmax),
        current_utilization(env),
        mean(loads) if loads else 0.0,
        pstdev(loads) if len(loads) > 1 else 0.0,
        float(env.get_state()["next_release_time"] or 0.0) / max_release,
    ]
    rule_bits: List[float] = []
    sched_rank = {job_id: idx for idx, job_id in enumerate(schedulable)}
    for idx, job_id in enumerate(choices):
        job = env.job_by_id[job_id]
        processing = [env.instance.processing_time[job_id][m] for m in job.candidate_machines]
        rule_bits.extend(
            [
                float(sched_rank.get(job_id, len(schedulable))) / max(1, len(schedulable) - 1),
                float(job.release_time) / max_release,
                max(0.0, now - job.release_time),
                mean_candidate_processing_time(env, job_id),
                estimated_completion_time(env, job_id),
                lookahead_score(env, job_id),
                1.0 if idx != 3 or ranker_available else 0.0,
            ]
        )
    return np.asarray(base + rule_bits, dtype=np.float32)


def select_action(model, state, device, greedy: bool = False):
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        dist, value = model.dist_value(state_t)
        action = torch.argmax(dist.probs, dim=-1) if greedy else dist.sample()
    return int(action.item()), float(dist.log_prob(action).item()), float(value.item())


def rollout_rule_policy(scenario: dict, rule_name: str, ranker_model=None, device: str = "cpu", top_k: int = 5):
    env = reset_env_for_scenario(scenario)
    counts = Counter()
    action_sequence = []
    start = time.perf_counter()
    while not env.is_done():
        job_id = _first_by_rule(env, rule_name, ranker_model=ranker_model, device=device, top_k=top_k)
        counts[rule_name] += 1
        env.step((job_id, choose_split_num(env, job_id)))
    runtime = time.perf_counter() - start
    return env, counts, runtime


def baseline_cmax(scenario: dict, method: str, ranker_model=None, device: str = "cpu", top_k: int = 5) -> float:
    rule = "FIFO" if method == "fifo" else "MLP_Ranker_soft_ce"
    env, _, _ = rollout_rule_policy(scenario, rule, ranker_model=ranker_model, device=device, top_k=top_k)
    return float(env.current_cmax)


def run_episode(model, scenario: dict, args, ranker_model, device, train_mode: bool):
    env = reset_env_for_scenario(scenario)
    buffer = PPOBuffer()
    counts = Counter()
    action_sequence = []
    step_reward_sum = 0.0
    old_util = current_utilization(env)
    while not env.is_done():
        choices = rule_choices(env, ranker_model=ranker_model, device=str(device), top_k=args.top_k)
        state = state_features(env, choices, ranker_model is not None)
        action, logprob, value = select_action(model, state, device, greedy=not train_mode)
        rule_name = RULE_NAMES[action]
        job_id = choices[action]
        env.step((job_id, choose_split_num(env, job_id)))
        new_util = current_utilization(env)
        if args.reward_mode == "util_plus_cmax":
            step_reward = new_util - old_util
        elif args.reward_mode == "env_cmax_delta":
            step_reward = -float(env.current_cmax)
        else:
            step_reward = 0.0
        old_util = new_util
        step_reward_sum += step_reward
        counts[rule_name] += 1
        action_sequence.append((action, rule_name))
        buffer.add(state, action, logprob, step_reward, value, env.is_done())

    base = baseline_cmax(scenario, args.baseline_method, ranker_model=ranker_model, device=str(device), top_k=args.top_k)
    agent_cmax = float(env.current_cmax)
    normalized_delta = (base - agent_cmax) / max(base, 1e-8)
    final_reward = float(args.reward_beta) * normalized_delta if args.reward_mode == "util_plus_cmax" else normalized_delta
    if buffer.rewards:
        buffer.rewards[-1] += final_reward
    total_reward = step_reward_sum + final_reward
    metrics = compute_metrics(env)
    return env, buffer, counts, action_sequence, {
        "episode_reward": total_reward,
        "episode_Cmax": agent_cmax,
        "machine_utilization": metrics["machine_utilization"],
        "step_reward_sum": step_reward_sum,
        "final_reward": final_reward,
        "total_reward": total_reward,
        "baseline_Cmax": base,
    }


def update(model, optimizer, buffer: PPOBuffer, args, device):
    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    batch = buffer.tensors(args.gamma, args.gae_lambda, device)
    n = len(buffer)
    stats = Counter()
    updates = 0
    for _ in range(args.update_epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, args.mini_batch_size):
            idx = order[start : start + args.mini_batch_size]
            dist, values = model.dist_value(batch["states"][idx])
            logprobs = dist.log_prob(batch["actions"][idx])
            ratio = torch.exp(logprobs - batch["old_logprobs"][idx])
            clipped = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * batch["advantages"][idx]
            policy_loss = -torch.min(ratio * batch["advantages"][idx], clipped).mean()
            value_loss = F.smooth_l1_loss(values, batch["returns"][idx])
            entropy = dist.entropy().mean()
            approx_kl = (batch["old_logprobs"][idx] - logprobs).mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stats.update(
                {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "approx_kl": float(approx_kl.item()),
                }
            )
            updates += 1
    return {key: stats[key] / max(1, updates) for key in ["policy_loss", "value_loss", "entropy", "approx_kl"]}


def evaluate_method(name: str, model, scenarios, args, ranker_model, device) -> dict:
    cmax_values = []
    reward_values = []
    runtimes = []
    valid = []
    for scenario in scenarios:
        if name == "HGCR-PPO":
            env, _, _, _, ep = run_episode(model, scenario, args, ranker_model, device, train_mode=False)
            reward_values.append(float(ep["total_reward"]))
            runtime = 0.0
        else:
            env, _, runtime = rollout_rule_policy(scenario, name, ranker_model=ranker_model, device=str(device), top_k=args.top_k)
            reward_values.append(0.0)
        cmax_values.append(float(env.current_cmax))
        runtimes.append(runtime)
        valid.append(1.0 if validate_schedule(env, scenario["instance"])["is_valid_schedule"] else 0.0)
    return {
        "Cmax_mean": mean(cmax_values),
        "Cmax_std": pstdev(cmax_values) if len(cmax_values) > 1 else 0.0,
        "reward_mean": mean(reward_values),
        "reward_std": pstdev(reward_values) if len(reward_values) > 1 else 0.0,
        "valid_schedule_rate": mean(valid),
        "runtime_mean": mean(runtimes),
    }


def evaluate_all(model, scenarios, args, ranker_model, device) -> List[dict]:
    methods = ["FIFO", "MLP_Ranker_soft_ce", "HGCR-PPO"]
    raw = {method: evaluate_method(method, model, scenarios, args, ranker_model, device) for method in methods}
    fifo = raw["FIFO"]["Cmax_mean"]
    mlp = raw["MLP_Ranker_soft_ce"]["Cmax_mean"]
    rows = []
    for method in methods:
        row = {
            "method": ACTION_DISPLAY_NAMES.get(method, method),
            "scenario_type": "dynamic",
            "size": args.size,
            "arrival_intensity": args.arrival_intensity,
            "carryover_ratio": args.carryover_ratio,
            **raw[method],
            "relative_to_FIFO": (fifo - raw[method]["Cmax_mean"]) / max(fifo, 1e-8),
            "relative_to_MLPRanker": (mlp - raw[method]["Cmax_mean"]) / max(mlp, 1e-8),
        }
        rows.append(row)
    return rows


def action_ratio_rows(counts: Counter) -> List[dict]:
    total = max(1, sum(counts.values()))
    return [{"rule_name": ACTION_DISPLAY_NAMES.get(rule, rule), "selection_count": counts[rule], "selection_ratio": counts[rule] / total} for rule in RULE_NAMES]


def action_stage_rows(action_history_rows: List[dict], args) -> List[dict]:
    stage_size = int(getattr(args, "action_stage_episodes", 1000))
    rows = []
    max_episode = max((int(row["episode"]) for row in action_history_rows), default=0)
    for start in range(1, max_episode + 1, stage_size):
        end = min(start + stage_size - 1, max_episode)
        bucket = [row for row in action_history_rows if start <= int(row["episode"]) <= end and not row["is_eval"]]
        total = max(1, len(bucket))
        counts = Counter(row["action_name"] for row in bucket)
        for name in ["FIFO", "GreedyECT", "Lookahead", "MLP-Ranker"]:
            rows.append(
                {
                    "stage_start_episode": start,
                    "stage_end_episode": end,
                    "size": args.size,
                    "seed": args.seed,
                    "arrival_intensity": args.arrival_intensity,
                    "carryover_ratio": args.carryover_ratio,
                    "reward_beta": args.reward_beta,
                    "action_name": name,
                    "action_count": counts[name],
                    "action_ratio": counts[name] / total,
                }
            )
    return rows


def run(args) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    ranker_model = load_ranker_or_none(args.ranker_ckpt, str(device))
    train_count = min(2, args.num_scenarios) if args.smoke_test else args.num_scenarios
    eval_count = min(2, args.eval_scenarios) if args.smoke_test else args.eval_scenarios
    scenarios = generate_dynamic_scenarios(
        args.size,
        "train",
        train_count,
        args.seed,
        args.arrival_intensity,
        args.carryover_ratio,
        processing_time_noise=args.processing_time_noise,
        machine_initial_load=args.machine_initial_load,
    )
    eval_scenarios = generate_dynamic_scenarios(
        args.size,
        "test",
        eval_count,
        args.seed + 999,
        args.arrival_intensity,
        args.carryover_ratio,
        processing_time_noise=args.processing_time_noise,
        machine_initial_load=args.machine_initial_load,
    )
    probe_env = reset_env_for_scenario(scenarios[0])
    probe_choices = rule_choices(probe_env, ranker_model=ranker_model, device=str(device), top_k=args.top_k)
    state_dim = len(state_features(probe_env, probe_choices, ranker_model is not None))
    model = RuleActorCritic(state_dim, len(RULE_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    run_id = make_run_id(args)
    paths = output_paths(args, run_id)

    buffer = PPOBuffer()
    train_rows = []
    curve_rows = []
    eval_history_rows = []
    action_history_rows = []
    action_counts = Counter()
    best_cmax = float("inf")
    best_state = None
    bad_evals = 0
    completed_episodes = 0
    early_stopped = False
    try:
        for episode in progress_iter(range(1, args.episodes + 1), desc="hgcr-dynamic-ppo train", total=args.episodes):
            completed_episodes = episode
            scenario = scenarios[(episode - 1) % len(scenarios)]
            _, ep_buffer, counts, actions, ep = run_episode(model, scenario, args, ranker_model, device, train_mode=True)
            for idx in range(len(ep_buffer)):
                buffer.add(ep_buffer.states[idx], ep_buffer.actions[idx], ep_buffer.logprobs[idx], ep_buffer.rewards[idx], ep_buffer.values[idx], ep_buffer.dones[idx])
            action_counts.update(counts)
            for decision_idx, (action_id, action_name) in enumerate(actions, start=1):
                action_history_rows.append(
                    {
                        "episode": episode,
                        "decision_index": decision_idx,
                        "size": args.size,
                        "seed": args.seed,
                        "arrival_intensity": args.arrival_intensity,
                        "carryover_ratio": args.carryover_ratio,
                        "reward_beta": args.reward_beta,
                        "action_id": action_id,
                        "action_name": ACTION_DISPLAY_NAMES.get(action_name, action_name),
                        "is_eval": False,
                    }
                )
            stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
            if len(buffer) >= args.batch_size or episode == args.episodes:
                stats = update(model, optimizer, buffer, args, device)
                buffer.clear()
            if episode % max(1, args.eval_interval) == 0 or episode == args.episodes:
                eval_rows = evaluate_all(model, eval_scenarios, args, ranker_model, device)
                hgcr_row = next(row for row in eval_rows if row["method"] == "HGCR-PPO")
                fifo_row = next(row for row in eval_rows if row["method"] == "FIFO")
                mlp_row = next(row for row in eval_rows if row["method"] == "MLP-Ranker")
                hgcr_cmax = hgcr_row["Cmax_mean"]
                if hgcr_cmax < best_cmax:
                    best_cmax = hgcr_cmax
                    best_state = deepcopy(model.state_dict())
                    bad_evals = 0
                else:
                    bad_evals += 1
                eval_history_rows.append(
                    {
                        "eval_step": len(eval_history_rows) + 1,
                        "episode": episode,
                        "size": args.size,
                        "seed": args.seed,
                        "arrival_intensity": args.arrival_intensity,
                        "carryover_ratio": args.carryover_ratio,
                        "reward_beta": args.reward_beta,
                        "eval_Cmax_mean": hgcr_row["Cmax_mean"],
                        "eval_Cmax_std": hgcr_row["Cmax_std"],
                        "eval_reward_mean": hgcr_row["reward_mean"],
                        "eval_reward_std": hgcr_row["reward_std"],
                        "best_so_far_Cmax": best_cmax,
                        "baseline_FIFO_Cmax": fifo_row["Cmax_mean"],
                        "baseline_MLPRanker_Cmax": mlp_row["Cmax_mean"],
                    }
                )
            row = {"episode": episode, **ep, **stats, "is_eval_step": bool(eval_history_rows and eval_history_rows[-1]["episode"] == episode)}
            train_rows.append(row)
            curve_rows.append({key: row[key] for key in CURVE_FIELDS})
            if bad_evals >= args.early_stop_patience and not args.smoke_test and not args.disable_early_stop:
                early_stopped = True
                print(f"Early stop at episode {episode}: no Cmax improvement for {bad_evals} eval checks.")
                break
    except Exception as exc:
        paths["manifest"].write_text(
            json.dumps(
                {
                    "stage": "G",
                    "experiment_family": "hgcr_dynamic_ppo",
                    "scenario_type": "dynamic_rolling",
                    "size": args.size,
                    "episodes": args.episodes,
                    "completed_episodes": completed_episodes,
                    "failed": True,
                    "failure_message": f"{type(exc).__name__}: {exc}",
                    "run_id": run_id,
                    "args": vars(args),
                    "python_version": sys.version,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise

    if best_state is not None:
        model.load_state_dict(best_state)
    eval_rows = evaluate_all(model, eval_scenarios, args, ranker_model, device)
    write_csv(train_rows, paths["train"], TRAIN_FIELDS)
    write_csv(eval_rows, paths["eval_summary"], EVAL_FIELDS)
    write_csv(action_ratio_rows(action_counts), paths["action_ratio"], ACTION_FIELDS)
    write_csv(curve_rows, paths["curve"], CURVE_FIELDS)
    write_csv(eval_history_rows, paths["eval_history"], EVAL_HISTORY_FIELDS)
    write_csv(action_history_rows, paths["action_history"], ACTION_HISTORY_FIELDS)
    write_csv(action_stage_rows(action_history_rows, args), paths["action_stage"], ACTION_STAGE_FIELDS)
    torch.save({"model_state_dict": model.state_dict(), "state_dim": state_dim, "rule_names": RULE_NAMES, "run_id": run_id}, paths["checkpoint"])
    paths["manifest"].write_text(
        json.dumps(
            {
                "stage": "G",
                "experiment_family": "hgcr_dynamic_ppo",
                "scenario_type": "dynamic_rolling",
                "size": args.size,
                "top_k": args.top_k,
                "episodes": args.episodes,
                "completed_episodes": completed_episodes,
                "early_stopped": early_stopped,
                "failed": False,
                "disable_early_stop": args.disable_early_stop,
                "seed": args.seed,
                "arrival_intensity": args.arrival_intensity,
                "carryover_ratio": args.carryover_ratio,
                "reward_mode": args.reward_mode,
                "reward_beta": args.reward_beta,
                "baseline_method": args.baseline_method,
                "output_dir": args.output_dir,
                "run_id": run_id,
                "method": "HGCR-Dynamic-PPO",
                "args": vars(args),
                "rule_names": RULE_NAMES,
                "state_dim": state_dim,
                "ranker_loaded": ranker_model is not None,
                "best_eval_Cmax_mean": best_cmax,
                "output_files": {key: str(value) for key, value in paths.items() if key != "run_dir"},
                "eval_history_path": str(paths["eval_history"]),
                "action_history_path": str(paths["action_history"]),
                "action_stage_summary_path": str(paths["action_stage"]),
                "python_version": sys.version,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved HGCR dynamic PPO run to {paths['run_dir']}")
    return paths["run_dir"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, default="small")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario_mode", choices=["dynamic"], default="dynamic")
    parser.add_argument("--arrival_intensity", choices=LEVELS, default="medium")
    parser.add_argument("--carryover_ratio", choices=LEVELS, default="medium")
    parser.add_argument("--processing_time_noise", type=float, choices=[0.0, 0.1, 0.2], default=0.0)
    parser.add_argument("--machine_initial_load", choices=LEVELS, default="low")
    parser.add_argument("--reward_mode", choices=REWARD_MODES, default="util_plus_cmax")
    parser.add_argument("--reward_beta", type=float, default=0.01)
    parser.add_argument("--baseline_method", choices=["fifo", "mlp_ranker_soft_ce"], default="fifo")
    parser.add_argument("--ranker_ckpt", default="checkpoints/stage_C/mlp_ranker/small_topk5_soft_ce/best.pt")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--disable_early_stop", action="store_true")
    parser.add_argument("--num_scenarios", type=int, default=200)
    parser.add_argument("--eval_scenarios", type=int, default=50)
    parser.add_argument("--action_stage_episodes", type=int, default=1000)
    parser.add_argument("--output_dir", default="data/results/stage_G/hgcr_dynamic_ppo")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
