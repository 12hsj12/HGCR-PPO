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
    create_legal_vs_selected_split_plot,
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
    "raw_episode_reward",
    "scaled_episode_reward",
    "train_episode_reward",
    "train_final_Cmax",
    "final_Cmax",
    "reference_Cmax",
    "relative_improvement_vs_reference",
    "train_average_completion_time",
    "train_average_waiting_time",
    "train_machine_utilization",
    "train_load_balance_std",
    "train_split_task_ratio",
    "train_total_split_count",
    "average_selected_split_num",
    "split_num_1_count",
    "split_num_2_count",
    "split_num_3_count",
    "split_num_4_count",
    "split_num_1_ratio",
    "split_num_2_ratio",
    "split_num_3_ratio",
    "split_num_4_ratio",
    "policy_loss",
    "value_loss",
    "entropy",
    "total_entropy",
    "job_entropy",
    "split_entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "grad_norm",
    "advantage_mean",
    "advantage_std",
    "return_mean",
    "return_std",
    "value_pred_mean",
    "value_target_mean",
    "raw_value_target_mean",
    "illegal_action_count",
    "legal_action_count",
    "action_mask_ratio",
    "obs_mean",
    "obs_std",
    "obs_min",
    "obs_max",
    "eval_Cmax_mean",
    "eval_Cmax_std",
    "eval_average_completion_time",
    "eval_average_waiting_time",
    "eval_machine_utilization",
    "eval_load_balance_std",
    "eval_split_task_ratio",
    "eval_total_split_count",
    "best_eval_Cmax",
    "best_model_path",
    "overfit_one_instance",
    "instance_index",
    "reference_heuristic",
    "FIFO_Cmax",
    "GreedyECT_Cmax",
    "Random_Cmax",
    "relative_improvement_vs_FIFO",
    "relative_improvement_vs_GreedyECT",
    "legal_split_1_count",
    "legal_split_2_count",
    "legal_split_3_count",
    "legal_split_4_count",
    "selected_split_1_count",
    "selected_split_2_count",
    "selected_split_3_count",
    "selected_split_4_count",
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


def _run_reference_heuristics(instance) -> List[Dict[str, float]]:
    from src.baselines.heuristics import run_heuristic

    rows = []
    for method in ["Random", "FIFO", "GreedyECT"]:
        result = run_heuristic(instance, method)
        rows.append({"method": method, **result.metrics})
    return rows


def _reference_value(rows: List[Dict[str, float]], method: str) -> float:
    for row in rows:
        if row["method"] == method:
            return float(row["Cmax_roll"])
    return float("nan")


def _run_label(config: PPOConfig) -> str:
    label = str(config.episodes)
    if config.overfit_one_instance:
        label += f"_overfit{config.instance_index}"
    return label


def _legal_split_counts(wrapper: VectorSchedulingWrapper) -> Dict[int, int]:
    masks = wrapper.get_split_masks()
    counts = {}
    for split_num in range(1, 5):
        column = split_num - 1
        counts[split_num] = int(masks[:, column].sum()) if column < masks.shape[1] else 0
    return counts


def _select_train_eval_instances(config: PPOConfig):
    if config.overfit_one_instance:
        instances = load_dataset(config.size, config.overfit_split)
        if config.instance_index < 0 or config.instance_index >= len(instances):
            raise IndexError(
                f"instance_index={config.instance_index} out of range for "
                f"{config.size}/{config.overfit_split}; available={len(instances)}"
            )
        instance = instances[config.instance_index]
        return [instance], [instance]
    return load_dataset(config.size, "train"), load_dataset(config.size, "test")


def train_ppo(config: PPOConfig) -> Dict[str, float]:
    _ensure_dirs()
    _set_seed(config.seed)
    generate_fixed_datasets(regenerate=False)
    train_instances, test_instances = _select_train_eval_instances(config)

    probe = VectorSchedulingWrapper(train_instances[0], config)
    agent = PPOAgent(probe.obs_dim, probe.action_dim, config)
    buffer = RolloutBuffer()
    rows = []
    best_eval = float("inf")
    run_label = _run_label(config)
    best_agent_path = Path("data/models") / f"ppo_{config.size}_{run_label}_best.pt"
    last_agent_path = Path("data/models") / f"ppo_{config.size}_{run_label}_last.pt"
    final_eval = None
    all_split_counts = {i: 0 for i in range(1, 5)}
    overfit_reference_rows = _run_reference_heuristics(train_instances[0]) if config.overfit_one_instance else []
    fifo_cmax = _reference_value(overfit_reference_rows, "FIFO") if config.overfit_one_instance else float("nan")
    greedy_cmax = _reference_value(overfit_reference_rows, "GreedyECT") if config.overfit_one_instance else float("nan")
    random_cmax = _reference_value(overfit_reference_rows, "Random") if config.overfit_one_instance else float("nan")

    progress = tqdm(
        range(1, config.episodes + 1),
        desc=f"PPO {config.size} {config.episodes}eps",
        dynamic_ncols=True,
    )
    for episode in progress:
        agent.train()
        instance = train_instances[(episode - 1) % len(train_instances)]
        wrapper = VectorSchedulingWrapper(instance, config)
        obs, _ = wrapper.reset(instance)
        obs_stats = wrapper.observation_stats()
        done = False
        raw_episode_reward = 0.0
        scaled_episode_reward = 0.0
        mask_ratios = []
        legal_split_totals = {idx: 0 for idx in range(1, 5)}

        while not done:
            masks = wrapper.get_policy_masks()
            step_legal = _legal_split_counts(wrapper)
            for key, value in step_legal.items():
                legal_split_totals[key] += value
            action, logprob, value = agent.select_action(obs, masks, greedy=False)
            if config.action_mode == "two_head":
                next_obs, reward, done, info, _ = wrapper.step_two_head(int(action[0]), int(action[1]))
            else:
                next_obs, reward, done, info, _ = wrapper.step(int(action))
            buffer.add(obs, action, logprob, reward, done, value, masks)
            raw_episode_reward += float(info.get("raw_reward", 0.0))
            scaled_episode_reward += reward
            mask_ratios.append(info["action_mask_ratio"])
            obs = next_obs

        update_stats = agent.update(buffer, episode=episode)
        metrics = compute_metrics(wrapper.env)
        split_dist = wrapper.split_distribution()
        total_split_choices = max(1, sum(split_dist.values()))
        for key, value in split_dist.items():
            all_split_counts[key] = all_split_counts.get(key, 0) + value
        avg_split = mean(wrapper.selected_split_nums) if wrapper.selected_split_nums else 0.0
        relative_improvement = (wrapper.reference_cmax - metrics["Cmax_roll"]) / wrapper.reference_cmax
        relative_vs_fifo = (fifo_cmax - metrics["Cmax_roll"]) / fifo_cmax if fifo_cmax == fifo_cmax and fifo_cmax > 0 else ""
        relative_vs_greedy = (greedy_cmax - metrics["Cmax_roll"]) / greedy_cmax if greedy_cmax == greedy_cmax and greedy_cmax > 0 else ""

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
            agent.train()

        row = {
            "episode": episode,
            "raw_episode_reward": raw_episode_reward,
            "scaled_episode_reward": scaled_episode_reward,
            "train_episode_reward": scaled_episode_reward,
            "train_final_Cmax": metrics["Cmax_roll"],
            "final_Cmax": metrics["Cmax_roll"],
            "reference_Cmax": wrapper.reference_cmax,
            "relative_improvement_vs_reference": relative_improvement,
            "train_average_completion_time": metrics["average_completion_time"],
            "train_average_waiting_time": metrics["average_waiting_time"],
            "train_machine_utilization": metrics["machine_utilization"],
            "train_load_balance_std": metrics["load_balance_std"],
            "train_split_task_ratio": metrics["split_task_ratio"],
            "train_total_split_count": metrics["total_split_count"],
            "average_selected_split_num": avg_split,
            "split_num_1_count": split_dist.get(1, 0),
            "split_num_2_count": split_dist.get(2, 0),
            "split_num_3_count": split_dist.get(3, 0),
            "split_num_4_count": split_dist.get(4, 0),
            "split_num_1_ratio": split_dist.get(1, 0) / total_split_choices,
            "split_num_2_ratio": split_dist.get(2, 0) / total_split_choices,
            "split_num_3_ratio": split_dist.get(3, 0) / total_split_choices,
            "split_num_4_ratio": split_dist.get(4, 0) / total_split_choices,
            "illegal_action_count": wrapper.illegal_action_count,
            "legal_action_count": wrapper.legal_action_count,
            "action_mask_ratio": mean(mask_ratios) if mask_ratios else 0.0,
            "best_eval_Cmax": best_eval if best_eval < float("inf") else "",
            "best_model_path": str(best_agent_path),
            "overfit_one_instance": config.overfit_one_instance,
            "instance_index": config.instance_index if config.overfit_one_instance else "",
            "reference_heuristic": config.reference_baseline,
            "FIFO_Cmax": fifo_cmax if fifo_cmax == fifo_cmax else "",
            "GreedyECT_Cmax": greedy_cmax if greedy_cmax == greedy_cmax else "",
            "Random_Cmax": random_cmax if random_cmax == random_cmax else "",
            "relative_improvement_vs_FIFO": relative_vs_fifo,
            "relative_improvement_vs_GreedyECT": relative_vs_greedy,
            "legal_split_1_count": legal_split_totals.get(1, 0),
            "legal_split_2_count": legal_split_totals.get(2, 0),
            "legal_split_3_count": legal_split_totals.get(3, 0),
            "legal_split_4_count": legal_split_totals.get(4, 0),
            "selected_split_1_count": split_dist.get(1, 0),
            "selected_split_2_count": split_dist.get(2, 0),
            "selected_split_3_count": split_dist.get(3, 0),
            "selected_split_4_count": split_dist.get(4, 0),
            **obs_stats,
            **update_stats,
            **eval_values,
        }
        rows.append(row)

        eval_msg = row["eval_Cmax_mean"] if row["eval_Cmax_mean"] != "" else "n/a"
        best_msg = f"{best_eval:.2f}" if best_eval < float("inf") else "n/a"
        progress.set_postfix(
            {
                "reward": f"{scaled_episode_reward:.3f}",
                "Cmax": f"{metrics['Cmax_roll']:.2f}",
                "eval": eval_msg if eval_msg == "n/a" else f"{float(eval_msg):.2f}",
                "best": best_msg,
            }
        )

    log_path = Path("data/logs") / f"ppo_train_{config.size}_{run_label}.csv"
    _write_csv(log_path, rows)
    agent.save(last_agent_path)
    if best_eval == float("inf"):
        agent.save(best_agent_path)
    gantt_prefix = f"data/results/ppo/gantt/gantt_ppo_{config.size}_{run_label}"
    final_eval = evaluate_agent(agent, test_instances, config, save_gantt_prefix=gantt_prefix)
    heuristic_rows = overfit_reference_rows if config.overfit_one_instance else evaluate_heuristics_fixed(config.size)
    eval_path = Path("data/results/ppo") / f"ppo_eval_{config.size}_{run_label}.csv"
    _write_eval(eval_path, final_eval, heuristic_rows)
    create_training_plots(log_path, config.size, run_label)
    create_split_distribution_plot(all_split_counts, config.size, run_label)
    create_legal_vs_selected_split_plot(log_path, config.size, run_label)
    create_comparison_plots(config.size, run_label, final_eval, heuristic_rows)
    if not config.overfit_one_instance:
        update_episode_sensitivity()
    print(f"saved log: {log_path}")
    print(f"saved best model: {best_agent_path}")
    print(f"saved last model: {last_agent_path}")
    print(f"saved eval: {eval_path}")
    return final_eval


def debug_ppo(config: PPOConfig) -> None:
    """One-episode PPO wiring diagnostic. Intended for manual remote use."""

    _set_seed(config.seed)
    generate_fixed_datasets(regenerate=False)
    if config.overfit_one_instance:
        instance = _select_train_eval_instances(config)[0][0]
    else:
        instance = load_dataset(config.size, "train")[0]
    wrapper = VectorSchedulingWrapper(instance, config)
    obs, _ = wrapper.reset(instance)
    agent = PPOAgent(wrapper.obs_dim, wrapper.action_dim, config)
    buffer = RolloutBuffer()
    masks = wrapper.get_policy_masks()
    action, logprob, value = agent.select_action(obs, masks, greedy=False)
    entropy_stats = agent.action_debug_stats(obs, masks, action)
    if config.action_mode == "two_head":
        selected_job_slot = int(action[0])
        selected_split_idx = int(action[1])
        next_obs, reward, done, info, _ = wrapper.step_two_head(selected_job_slot, selected_split_idx)
        selected_job = wrapper.jobs[selected_job_slot].job_id
        selected_split_num = selected_split_idx + 1
    else:
        selected_job, selected_split_num = wrapper.decode_action(int(action))
        next_obs, reward, done, info, _ = wrapper.step(int(action))
    first_info = dict(info)
    first_reward = reward
    buffer.add(obs, action, logprob, reward, done, value, masks)

    while not wrapper.env.is_done():
        masks = wrapper.get_policy_masks()
        action, logprob, value = agent.select_action(next_obs, masks, greedy=False)
        if config.action_mode == "two_head":
            current_obs = next_obs
            next_obs, reward, done, info, _ = wrapper.step_two_head(int(action[0]), int(action[1]))
        else:
            current_obs = next_obs
            next_obs, reward, done, info, _ = wrapper.step(int(action))
        buffer.add(current_obs, action, logprob, reward, done, value, masks)

    batch = buffer.compute_returns_advantages(
        config.gamma,
        config.gae_lambda,
        use_return_normalization=config.use_return_normalization,
    )

    metrics = compute_metrics(wrapper.env)
    split_dist = wrapper.split_distribution()
    total_split_choices = max(1, sum(split_dist.values()))
    print(f"observation shape: {obs.shape}")
    print(
        f"obs_mean/std/min/max: {obs.mean():.6f}/{obs.std():.6f}/"
        f"{obs.min():.6f}/{obs.max():.6f}"
    )
    if np.max(np.abs(obs)) > 20:
        print("WARNING: observation contains absolute values greater than 20.")
    print(f"job_mask shape: {masks['job'].shape}")
    print(f"split_mask shape: {masks['split'].shape}")
    print(f"legal job count: {int(masks['job'].sum())}")
    print(f"sample action: {action}")
    print(f"selected job_id: {selected_job}")
    print(f"selected split_num: {selected_split_num}")
    print(f"raw_reward: {first_info.get('raw_reward', 0.0):.6f}")
    print(f"scaled_reward: {first_reward:.6f}")
    print(f"reference_Cmax: {wrapper.reference_cmax:.6f}")
    print(f"final_Cmax: {metrics['Cmax_roll']:.6f}")
    print(f"advantage mean/std: {batch.stats['advantage_mean']:.6f}/{batch.stats['advantage_std']:.6f}")
    print(f"return mean/std: {batch.stats['return_mean']:.6f}/{batch.stats['return_std']:.6f}")
    print(f"job_entropy / split_entropy: {entropy_stats['job_entropy']:.6f}/{entropy_stats['split_entropy']:.6f}")
    print(
        "split_num ratios: "
        f"1={split_dist.get(1, 0) / total_split_choices:.3f}, "
        f"2={split_dist.get(2, 0) / total_split_choices:.3f}, "
        f"3={split_dist.get(3, 0) / total_split_choices:.3f}, "
        f"4={split_dist.get(4, 0) / total_split_choices:.3f}"
    )
