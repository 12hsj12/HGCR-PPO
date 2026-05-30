"""Evaluation utilities for PPO and fixed-instance heuristic baselines."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

from configs.ppo_config import PPOConfig
from dataset_manager import load_dataset
from ppo.state_encoder import VectorSchedulingWrapper
from src.baselines.heuristics import POLICIES, run_heuristic
from src.evaluation.metrics import compute_metrics


METRIC_KEYS = [
    "Cmax_roll",
    "average_completion_time",
    "average_waiting_time",
    "machine_utilization",
    "load_balance_std",
    "split_task_ratio",
    "total_split_count",
]


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    out = {}
    for key in METRIC_KEYS:
        vals = [row[key] for row in rows]
        out[key] = mean(vals)
        out[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
    return out


def evaluate_agent(agent, instances, config: PPOConfig, save_gantt_prefix: str | None = None) -> Dict[str, float]:
    agent.eval()
    metrics_rows = []
    start = time.perf_counter()
    best_env = None
    last_env = None
    best_cmax = float("inf")

    for instance in instances:
        wrapper = VectorSchedulingWrapper(instance, config)
        obs, _ = wrapper.reset(instance)
        done = False
        while not done:
            masks = wrapper.get_policy_masks()
            action, _, _ = agent.select_action(obs, masks, greedy=True)
            if config.policy_mode == "order_only":
                obs, _, done, _, _ = wrapper.step_order_only(int(action[0]))
            elif config.action_mode == "two_head":
                obs, _, done, _, _ = wrapper.step_two_head(int(action[0]), int(action[1]))
            else:
                obs, _, done, _, _ = wrapper.step(int(action))
        metrics = compute_metrics(wrapper.env)
        metrics_rows.append(metrics)
        last_env = wrapper.env
        if metrics["Cmax_roll"] < best_cmax:
            best_cmax = metrics["Cmax_roll"]
            best_env = wrapper.env

    elapsed = time.perf_counter() - start
    agg = _aggregate(metrics_rows)
    result = {
        "test_Cmax_mean": agg["Cmax_roll"],
        "test_Cmax_std": agg["Cmax_roll_std"],
        "test_average_completion_time": agg["average_completion_time"],
        "test_average_waiting_time": agg["average_waiting_time"],
        "test_machine_utilization": agg["machine_utilization"],
        "test_load_balance_std": agg["load_balance_std"],
        "test_split_task_ratio": agg["split_task_ratio"],
        "test_total_split_count": agg["total_split_count"],
        "inference_time_per_instance": elapsed / max(1, len(instances)),
    }
    if save_gantt_prefix and best_env and last_env:
        best_env.render_gantt(f"{save_gantt_prefix}_best.png")
        last_env.render_gantt(f"{save_gantt_prefix}_last.png")
    return result


def evaluate_heuristics_fixed(size: str, output_path: str = "data/results/ppo/heuristic_baselines_fixed_instances.csv") -> List[Dict[str, float]]:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    instances = load_dataset(size, "test")
    rows = []
    for heuristic in POLICIES:
        metric_rows = [run_heuristic(instance, heuristic).metrics for instance in instances]
        agg = _aggregate(metric_rows)
        rows.append(
            {
                "size": size,
                "method": heuristic,
                "Cmax_roll": agg["Cmax_roll"],
                "average_completion_time": agg["average_completion_time"],
                "average_waiting_time": agg["average_waiting_time"],
                "machine_utilization": agg["machine_utilization"],
                "load_balance_std": agg["load_balance_std"],
                "split_task_ratio": agg["split_task_ratio"],
                "total_split_count": agg["total_split_count"],
            }
        )

    existing = []
    path = Path(output_path)
    if path.exists():
        with path.open("r", newline="") as f:
            existing = [
                row
                for row in csv.DictReader(f)
                if row.get("size") != size and "method" in row
            ]
    all_rows = existing + rows
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    return rows
