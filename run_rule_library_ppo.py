"""Experience-guided Rule Library PPO for HGCR-PPO Stage F2-3."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import uuid
from collections import Counter
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

from candidate_generator import generate_candidates
from instance_manager import SIZES, SPLITS, ensure_fixed_dataset, load_fixed_instances
from schedule_validator import validate_schedule
from src.baselines.heuristics import choose_split_num
from src.envs.rolling_scheduling_env import RollingSchedulingEnv
from src.evaluation.metrics import compute_metrics
from src.rules.experience_rule_library import ExperienceRuleLibrary, RULE_NAMES
from train_rule_library_bc import RuleBCSelector
from utils.experiment_io import progress_iter, write_csv


METHOD = "RuleLibraryPPO"
TRAIN_FIELDS = ["episode", "reward", "final_cmax", "baseline_cmax", "delta_improvement", "policy_loss", "value_loss", "entropy", "kl_to_bc", "eval_delta_improvement_mean", "is_best"]
EVAL_FIELDS = ["run_id", "method", "checkpoint_type", "instance_id", "size", "split", "top_k", "episodes", "seed", "final_cmax", "baseline_Cmax", "delta_improvement", "normalized_delta_improvement", "base_rule_ratio", "non_base_rule_ratio", "fallback_count", "is_valid_schedule"]
SUMMARY_FIELDS = ["run_id", "method", "checkpoint_type", "size", "split", "top_k", "episodes", "seed", "Cmax_mean", "Cmax_std", "baseline_Cmax_mean", "delta_improvement_mean", "delta_improvement_std", "normalized_delta_improvement_mean", "base_rule_ratio_mean", "non_base_rule_ratio_mean", "fallback_count_mean", "valid_schedule_rate"]
ACTION_FIELDS = ["rule_name", "rule_count", "rule_ratio", "mean_proxy_score", "fallback_count", "fallback_ratio"]


@dataclass
class PPOBuffer:
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

    def tensors(self, gamma: float, gae_lambda: float, device) -> Dict[str, torch.Tensor]:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray([*self.values, 0.0], dtype=np.float32)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for idx in reversed(range(len(rewards))):
            delta = rewards[idx] + gamma * values[idx + 1] * (1.0 - dones[idx]) - values[idx]
            gae = delta + gamma * gae_lambda * (1.0 - dones[idx]) * gae
            adv[idx] = gae
        returns = adv + values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) else adv
        return {
            "states": torch.tensor(np.asarray(self.states), dtype=torch.float32, device=device),
            "actions": torch.tensor(self.actions, dtype=torch.long, device=device),
            "masks": torch.tensor(np.asarray(self.masks), dtype=torch.bool, device=device),
            "old_logprobs": torch.tensor(self.logprobs, dtype=torch.float32, device=device),
            "returns": torch.tensor(returns, dtype=torch.float32, device=device),
            "advantages": torch.tensor(adv, dtype=torch.float32, device=device),
        }


class RuleLibraryActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.actor_body = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def logits_value(self, states):
        return self.actor_head(self.actor_body(states)), self.critic(states).squeeze(-1)

    def dist_value(self, states, masks):
        logits, value = self.logits_value(states)
        return torch.distributions.Categorical(logits=logits.masked_fill(~masks.bool(), -1e9)), value


def make_run_id(args) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"RuleLibPPO_{args.size}_{args.split}_k{args.top_k}_e{args.episodes}_s{args.seed}_{args.reward_mode}_{stamp}_{uuid.uuid4().hex[:8]}"


def output_paths(args, run_id: str) -> Dict[str, Path]:
    run_dir = Path(args.output_dir) / "runs" / run_id
    ckpt_dir = Path("checkpoints/stage_F/rule_library_ppo") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir.mkdir(parents=True, exist_ok=False)
    return {
        "run_dir": run_dir,
        "best": ckpt_dir / "best.pt",
        "last": ckpt_dir / "last.pt",
        "train": run_dir / f"train_log__{run_id}.csv",
        "eval_last": run_dir / f"eval_last__{run_id}.csv",
        "eval_best": run_dir / f"eval_best__{run_id}.csv",
        "summary_last": run_dir / f"eval_summary_last__{run_id}.csv",
        "summary_best": run_dir / f"eval_summary_best__{run_id}.csv",
        "actions": run_dir / f"rule_action_stats__{run_id}.csv",
        "manifest": run_dir / f"manifest__{run_id}.json",
    }


def load_bc_model(path: str | None, state_dim: int, device):
    if not path:
        return None
    checkpoint = torch.load(path, map_location=device)
    model = RuleBCSelector(int(checkpoint.get("input_dim", state_dim)), action_dim=len(RULE_NAMES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def init_from_bc(model, bc_model: RuleBCSelector | None, keep_base_bias: float) -> None:
    if bc_model is None:
        with torch.no_grad():
            model.actor_head.bias.zero_()
            model.actor_head.bias[0] = float(keep_base_bias)
        return
    # Architecture differs slightly; copy the final classifier when dimensions match.
    with torch.no_grad():
        if bc_model.net[-1].weight.shape == model.actor_head.weight.shape:
            model.actor_head.weight.copy_(bc_model.net[-1].weight)
            model.actor_head.bias.copy_(bc_model.net[-1].bias)


def select_action(model, state, mask, device, greedy=False):
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
    with torch.no_grad():
        dist, value = model.dist_value(state_t, mask_t)
        action = torch.argmax(dist.probs, dim=-1) if greedy else dist.sample()
    return int(action.item()), float(dist.log_prob(action).item()), float(value.item())


def baseline_cmax(instance, args, library) -> float:
    env = RollingSchedulingEnv(instance)
    env.reset(instance)
    while not env.is_done():
        candidates = generate_candidates(env, candidate_mode="hybrid_topk", top_k=args.top_k, fallback_to_all=True)
        recs = library.recommend(env, candidates)
        job_id, _ = library.choose_job_or_fallback(recs, 0)
        env.step((job_id, choose_split_num(env, job_id)))
    return float(env.current_cmax)


def run_episode(model, instance, args, library, device, train_mode: bool):
    env = RollingSchedulingEnv(instance)
    env.reset(instance)
    buffer = PPOBuffer()
    counts = Counter()
    proxy_sum = Counter()
    fallback = 0
    while not env.is_done():
        candidates = generate_candidates(env, candidate_mode="hybrid_topk", top_k=args.top_k, fallback_to_all=True)
        recs = library.recommend(env, candidates)
        mask = library.action_mask(recs)
        state = library.state_features(env, candidates, recs)
        action, logprob, value = select_action(model, state, mask, device, greedy=not train_mode)
        job_id, executed_rule = library.choose_job_or_fallback(recs, action)
        if executed_rule != action:
            fallback += 1
        counts[RULE_NAMES[executed_rule]] += 1
        proxy_sum[RULE_NAMES[executed_rule]] += recs[executed_rule].score_or_proxy
        env.step((job_id, choose_split_num(env, job_id)))
        buffer.add(state, action, mask, logprob, 0.0, value, env.is_done())
    base = baseline_cmax(instance, args, library)
    final = float(env.current_cmax)
    non_base_ratio = 1.0 - counts[RULE_NAMES[0]] / max(1, sum(counts.values()))
    delta = base - final
    normalized = delta / max(base, 1e-8) * 100.0
    reward = normalized - args.override_penalty * non_base_ratio
    if buffer.rewards:
        buffer.rewards[-1] = reward
    return env, buffer, counts, proxy_sum, fallback, base, delta, normalized, reward


def update(model, bc_model, optimizer, buffer, args, device, freeze_actor: bool):
    batch = buffer.tensors(args.gamma, args.gae_lambda, device)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl_to_bc": 0.0}
    updates = 0
    for _ in range(args.update_epochs):
        dist, values = model.dist_value(batch["states"], batch["masks"])
        logprobs = dist.log_prob(batch["actions"])
        ratio = torch.exp(logprobs - batch["old_logprobs"])
        policy_loss = -torch.min(ratio * batch["advantages"], torch.clamp(ratio, 0.8, 1.2) * batch["advantages"]).mean()
        value_loss = F.smooth_l1_loss(values, batch["returns"])
        entropy = dist.entropy().mean()
        kl = torch.zeros((), device=device)
        if bc_model is not None:
            with torch.no_grad():
                bc_logits = bc_model(batch["states"]).masked_fill(~batch["masks"], -1e9)
                bc_probs = torch.softmax(bc_logits, dim=1)
            kl = torch.distributions.kl_divergence(torch.distributions.Categorical(probs=bc_probs), dist).mean()
        if freeze_actor:
            loss = value_loss * args.value_coef
        else:
            loss = policy_loss + value_loss * args.value_coef - args.entropy_coef * entropy + args.kl_to_bc_coef * kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stats["policy_loss"] += float(policy_loss.item())
        stats["value_loss"] += float(value_loss.item())
        stats["entropy"] += float(entropy.item())
        stats["kl_to_bc"] += float(kl.item())
        updates += 1
    return {k: v / max(1, updates) for k, v in stats.items()}


def evaluate_policy(model, instances, args, library, device, run_id, checkpoint_type):
    rows = []
    action_counts = Counter()
    proxy_sums = Counter()
    fallback_total = 0
    for instance in instances:
        env, _, counts, proxy_sum, fallback, base, delta, normalized, _ = run_episode(model, instance, args, library, device, False)
        metrics = compute_metrics(env)
        valid = validate_schedule(env, instance)
        total = max(1, sum(counts.values()))
        rows.append(
            {
                "run_id": run_id,
                "method": METHOD,
                "checkpoint_type": checkpoint_type,
                "instance_id": getattr(instance, "instance_id", getattr(instance, "name", "")),
                "size": args.size,
                "split": args.split,
                "top_k": args.top_k,
                "episodes": args.episodes,
                "seed": args.seed,
                "final_cmax": metrics["Cmax_roll"],
                "baseline_Cmax": base,
                "delta_improvement": delta,
                "normalized_delta_improvement": normalized,
                "base_rule_ratio": counts[RULE_NAMES[0]] / total,
                "non_base_rule_ratio": 1.0 - counts[RULE_NAMES[0]] / total,
                "fallback_count": fallback,
                "is_valid_schedule": valid["is_valid_schedule"],
            }
        )
        action_counts.update(counts)
        proxy_sums.update(proxy_sum)
        fallback_total += fallback
    return rows, action_rows(action_counts, proxy_sums, fallback_total)


def action_rows(counts, proxy_sums, fallback_total):
    total = max(1, sum(counts.values()))
    return [
        {
            "rule_name": rule,
            "rule_count": counts[rule],
            "rule_ratio": counts[rule] / total,
            "mean_proxy_score": proxy_sums[rule] / max(1, counts[rule]),
            "fallback_count": fallback_total if rule == RULE_NAMES[0] else 0,
            "fallback_ratio": fallback_total / total if rule == RULE_NAMES[0] else 0.0,
        }
        for rule in RULE_NAMES
    ]


def summarize(rows, args, run_id, checkpoint_type):
    values = {k: [float(row[k]) for row in rows] for k in ["final_cmax", "baseline_Cmax", "delta_improvement", "normalized_delta_improvement", "base_rule_ratio", "non_base_rule_ratio", "fallback_count"]}
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
            "Cmax_mean": mean(values["final_cmax"]),
            "Cmax_std": pstdev(values["final_cmax"]) if len(rows) > 1 else 0.0,
            "baseline_Cmax_mean": mean(values["baseline_Cmax"]),
            "delta_improvement_mean": mean(values["delta_improvement"]),
            "delta_improvement_std": pstdev(values["delta_improvement"]) if len(rows) > 1 else 0.0,
            "normalized_delta_improvement_mean": mean(values["normalized_delta_improvement"]),
            "base_rule_ratio_mean": mean(values["base_rule_ratio"]),
            "non_base_rule_ratio_mean": mean(values["non_base_rule_ratio"]),
            "fallback_count_mean": mean(values["fallback_count"]),
            "valid_schedule_rate": mean([1.0 if row["is_valid_schedule"] else 0.0 for row in rows]),
        }
    ]


def save_ckpt(path, model, args, run_id, state_dim):
    if path.exists():
        raise FileExistsError(path)
    torch.save({"model_state_dict": model.state_dict(), "state_dim": state_dim, "rule_names": RULE_NAMES, "metadata": {"run_id": run_id, "args": vars(args)}}, path)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    ensure_fixed_dataset([args.size], [args.split])
    instances = load_fixed_instances(args.size, args.split)
    if args.smoke_test:
        instances = instances[:1]
    library = ExperienceRuleLibrary(ranker_ckpt=args.ranker_ckpt, device=str(device))
    probe_env = RollingSchedulingEnv(instances[0])
    probe_env.reset(instances[0])
    probe_candidates = generate_candidates(probe_env, candidate_mode="hybrid_topk", top_k=args.top_k, fallback_to_all=True)
    probe_recs = library.recommend(probe_env, probe_candidates)
    state_dim = len(library.state_features(probe_env, probe_candidates, probe_recs))
    model = RuleLibraryActorCritic(state_dim, len(RULE_NAMES)).to(device)
    bc_model = load_bc_model(args.bc_init_ckpt, state_dim, device)
    init_from_bc(model, bc_model, args.keep_base_bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    run_id = make_run_id(args)
    paths = output_paths(args, run_id)
    train_rows = []
    best_score = -float("inf")
    best_state = None
    for episode in progress_iter(range(1, args.episodes + 1), desc="rule-library-ppo train", total=args.episodes):
        instance = instances[(episode - 1) % len(instances)]
        _, buffer, _, _, _, base, delta, _, reward = run_episode(model, instance, args, library, device, True)
        stats = update(model, bc_model, optimizer, buffer, args, device, episode <= args.freeze_actor_episodes)
        eval_delta = ""
        is_best = False
        if episode % max(1, args.eval_interval) == 0 or episode == args.episodes:
            eval_rows, _ = evaluate_policy(model, instances, args, library, device, run_id, "interval")
            eval_delta = summarize(eval_rows, args, run_id, "interval")[0]["delta_improvement_mean"]
            if float(eval_delta) > best_score:
                best_score = float(eval_delta)
                best_state = deepcopy(model.state_dict())
                is_best = True
        train_rows.append({"episode": episode, "reward": reward, "final_cmax": base - delta, "baseline_cmax": base, "delta_improvement": delta, "eval_delta_improvement_mean": eval_delta, "is_best": is_best, **stats})
    last_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    save_ckpt(paths["best"], model, args, run_id, state_dim)
    best_rows, best_actions = evaluate_policy(model, instances, args, library, device, run_id, "best")
    model.load_state_dict(last_state)
    save_ckpt(paths["last"], model, args, run_id, state_dim)
    last_rows, last_actions = evaluate_policy(model, instances, args, library, device, run_id, "last")

    write_csv(train_rows, paths["train"], TRAIN_FIELDS)
    write_csv(last_rows, paths["eval_last"], EVAL_FIELDS)
    write_csv(best_rows, paths["eval_best"], EVAL_FIELDS)
    write_csv(summarize(last_rows, args, run_id, "last"), paths["summary_last"], SUMMARY_FIELDS)
    write_csv(summarize(best_rows, args, run_id, "best"), paths["summary_best"], SUMMARY_FIELDS)
    write_csv(best_actions or last_actions, paths["actions"], ACTION_FIELDS)
    paths["manifest"].write_text(json.dumps({"run_id": run_id, "args": vars(args), "python_version": sys.version}, indent=2), encoding="utf-8")
    print(f"Saved Rule Library PPO run to {paths['run_dir']}")


def make_run_id(args) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"RuleLibPPO_{args.size}_{args.split}_k{args.top_k}_e{args.episodes}_s{args.seed}_{args.reward_mode}_{stamp}_{uuid.uuid4().hex[:8]}"


def output_paths(args, run_id):
    run_dir = Path(args.output_dir) / "runs" / run_id
    ckpt_dir = Path("checkpoints/stage_F/rule_library_ppo") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir.mkdir(parents=True, exist_ok=False)
    return {
        "run_dir": run_dir,
        "best": ckpt_dir / "best.pt",
        "last": ckpt_dir / "last.pt",
        "train": run_dir / f"train_log__{run_id}.csv",
        "eval_last": run_dir / f"eval_last__{run_id}.csv",
        "eval_best": run_dir / f"eval_best__{run_id}.csv",
        "summary_last": run_dir / f"eval_summary_last__{run_id}.csv",
        "summary_best": run_dir / f"eval_summary_best__{run_id}.csv",
        "actions": run_dir / f"rule_action_stats__{run_id}.csv",
        "manifest": run_dir / f"manifest__{run_id}.json",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, default="small")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ranker_ckpt", default="checkpoints/stage_C/mlp_ranker/small_topk5_soft_ce/best.pt")
    parser.add_argument("--bc_init_ckpt", default=None)
    parser.add_argument("--baseline_method", choices=["mlp_ranker_soft_ce"], default="mlp_ranker_soft_ce")
    parser.add_argument("--reward_mode", choices=["conservative_final_delta"], default="conservative_final_delta")
    parser.add_argument("--override_penalty", type=float, default=0.03)
    parser.add_argument("--keep_base_bias", type=float, default=2.0)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--freeze_actor_episodes", type=int, default=20)
    parser.add_argument("--kl_to_bc_coef", type=float, default=0.01)
    parser.add_argument("--output_dir", default="data/results/stage_F/rule_library_ppo")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--update_epochs", type=int, default=4)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
