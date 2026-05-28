"""PPO training entry point for fixed TSG-PPO instances."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from configs.ppo_config import PPOConfig
from dataset_manager import generate_fixed_datasets, load_dataset
from ppo.evaluate_ppo import evaluate_agent, evaluate_heuristics_fixed
from ppo.plot_training import (
    create_comparison_plots,
    create_split_distribution_plot,
    create_training_plots,
    update_episode_sensitivity,
)
from ppo.ppo_agent import PPOAgent
from ppo.rollout_buffer import RolloutBuffer
from ppo.state_encoder import VectorSchedulingWrapper
from src.evaluation.metrics import compute_metrics


LOG_FIELDS = [
    "episode",
    "train_episode_reward",
    "train_final_Cmax",
    "train_average_completion_time",
    "train_average_waiting_time",
    "train_machine_utilization",
    "train_load_balance_std",
    "train_split_task_ratio",
    "train_total_split_count",
    "average_selected_split_num",
    "policy_loss",
    "value_loss",
    "entropy",
    "illegal_action_count",
    "legal_action_count",
    "action_mask_ratio",
    "split_num_1_count",
    "split_num_2_count",
    "split_num_3_count",
    "split_num_4_count",
    "eval_Cmax_mean",
    "eval_Cmax_std",
    "eval_average_completion_time",
    "eval_average_waiting_time",
    "eval_machine_utilization",
    "eval_load_balance_std",
    "eval_split_task_ratio",
    "eval_total_split_count",
]


def _ensure_dirs() -> None:
    for path in [
        "data/models",
        "data/logs",
        "data/results/ppo",
        "data/results/ppo/plots",
        "data/results/ppo/gantt",
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _empty_eval() -> Dict[str, str]:
    return {
        "eval_Cmax_mean": "",
        "eval_Cmax_std": "",
        "eval_average_completion_time": "",
        "eval_average_waiting_time": "",
        "eval_machine_utilization": "",
        "eval_load_balance_std": "",
        "eval_split_task_ratio": "",
        "eval_total_split_count": "",
    }


def _write_csv(path: str | Path, rows: List[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_eval(path: str | Path, metrics: Dict[str, float], heuristic_rows: List[Dict[str, float]]) -> None:
    fields = list(metrics.keys()) + [f"improvement_vs_{row['method']}_pct" for row in heuristic_rows]
    row = dict(metrics)
    for baseline in heuristic_rows:
        base = float(baseline["Cmax_roll"])
        row[f"improvement_vs_{baseline['method']}_pct"] = (base - metrics["test_Cmax_mean"]) / base * 100.0
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def train_ppo(config: PPOConfig) -> Dict[str, float]:
    _ensure_dirs()
    _set_seed(config.seed)
    generate_fixed_datasets(regenerate=False)
    train_instances = load_dataset(config.size, "train")
    test_instances = load_dataset(config.size, "test")

    probe = VectorSchedulingWrapper(train_instances[0], config)
    agent = PPOAgent(probe.obs_dim, probe.action_dim, config)
    buffer = RolloutBuffer()
    rows = []
    best_eval = float("inf")
    best_agent_path = Path("data/models") / f"ppo_{config.size}_{config.episodes}.pt"
    final_eval = None
    all_split_counts = {i: 0 for i in range(1, config.limits["max_split"] + 1)}

    progress = tqdm(
        range(1, config.episodes + 1),
        desc=f"PPO {config.size} {config.episodes}eps",
        dynamic_ncols=True,
    )
    for episode in progress:
        instance = train_instances[(episode - 1) % len(train_instances)]
        wrapper = VectorSchedulingWrapper(instance, config)
        obs, mask = wrapper.reset(instance)
        done = False
        episode_reward = 0.0
        mask_ratios = []

        while not done:
            action, logprob, value = agent.select_action(obs, mask, greedy=False)
            next_obs, reward, done, info, next_mask = wrapper.step(action)
            buffer.add(obs, action, logprob, reward, done, value, mask)
            episode_reward += reward
            mask_ratios.append(info["action_mask_ratio"])
            obs, mask = next_obs, next_mask

        update_stats = agent.update(buffer)
        metrics = compute_metrics(wrapper.env)
        split_dist = wrapper.split_distribution()
        for key, value in split_dist.items():
            all_split_counts[key] = all_split_counts.get(key, 0) + value
        avg_split = mean(wrapper.selected_split_nums) if wrapper.selected_split_nums else 0.0

        eval_values = _empty_eval()
        if episode % config.eval_interval == 0 or episode == config.episodes:
            final_eval = evaluate_agent(agent, test_instances, config)
            eval_values = {
                "eval_Cmax_mean": final_eval["test_Cmax_mean"],
                "eval_Cmax_std": final_eval["test_Cmax_std"],
                "eval_average_completion_time": final_eval["test_average_completion_time"],
                "eval_average_waiting_time": final_eval["test_average_waiting_time"],
                "eval_machine_utilization": final_eval["test_machine_utilization"],
                "eval_load_balance_std": final_eval["test_load_balance_std"],
                "eval_split_task_ratio": final_eval["test_split_task_ratio"],
                "eval_total_split_count": final_eval["test_total_split_count"],
            }
            if final_eval["test_Cmax_mean"] < best_eval:
                best_eval = final_eval["test_Cmax_mean"]
                agent.save(best_agent_path)

        row = {
            "episode": episode,
            "train_episode_reward": episode_reward,
            "train_final_Cmax": metrics["Cmax_roll"],
            "train_average_completion_time": metrics["average_completion_time"],
            "train_average_waiting_time": metrics["average_waiting_time"],
            "train_machine_utilization": metrics["machine_utilization"],
            "train_load_balance_std": metrics["load_balance_std"],
            "train_split_task_ratio": metrics["split_task_ratio"],
            "train_total_split_count": metrics["total_split_count"],
            "average_selected_split_num": avg_split,
            "policy_loss": update_stats["policy_loss"],
            "value_loss": update_stats["value_loss"],
            "entropy": update_stats["entropy"],
            "illegal_action_count": wrapper.illegal_action_count,
            "legal_action_count": wrapper.legal_action_count,
            "action_mask_ratio": mean(mask_ratios) if mask_ratios else 0.0,
            "split_num_1_count": split_dist.get(1, 0),
            "split_num_2_count": split_dist.get(2, 0),
            "split_num_3_count": split_dist.get(3, 0),
            "split_num_4_count": split_dist.get(4, 0),
            **eval_values,
        }
        rows.append(row)

        eval_msg = row["eval_Cmax_mean"] if row["eval_Cmax_mean"] != "" else "n/a"
        best_msg = f"{best_eval:.2f}" if best_eval < float("inf") else "n/a"
        progress.set_postfix(
            {
                "reward": f"{episode_reward:.2f}",
                "Cmax": f"{metrics['Cmax_roll']:.2f}",
                "eval": eval_msg if eval_msg == "n/a" else f"{float(eval_msg):.2f}",
                "best": best_msg,
            }
        )

    log_path = Path("data/logs") / f"ppo_train_{config.size}_{config.episodes}.csv"
    _write_csv(log_path, rows)
    agent.save(best_agent_path)
    gantt_prefix = f"data/results/ppo/gantt/gantt_ppo_{config.size}_{config.episodes}"
    final_eval = evaluate_agent(agent, test_instances, config, save_gantt_prefix=gantt_prefix)
    heuristic_rows = evaluate_heuristics_fixed(config.size)
    eval_path = Path("data/results/ppo") / f"ppo_eval_{config.size}_{config.episodes}.csv"
    _write_eval(eval_path, final_eval, heuristic_rows)
    create_training_plots(log_path, config.size, config.episodes)
    create_split_distribution_plot(all_split_counts, config.size, config.episodes)
    create_comparison_plots(config.size, config.episodes, final_eval, heuristic_rows)
    update_episode_sensitivity()
    print(f"saved log: {log_path}")
    print(f"saved model: {best_agent_path}")
    print(f"saved eval: {eval_path}")
    return final_eval
