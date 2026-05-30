"""Vector observation encoder and PPO action/reward wrapper."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from configs.ppo_config import PPOConfig
from src.baselines.heuristics import run_heuristic
from src.envs.rolling_scheduling_env import RollingSchedulingEnv


PROCESS_TYPES = ["sl", "cu", "co"]
_REFERENCE_CMAX_CACHE: Dict[tuple[str, str], float] = {}


class VectorSchedulingWrapper:
    """Padded vector-state interface for PPO.

    The original environment remains untouched. This wrapper handles fixed-size
    observations, action masks, two-head action adaptation, and PPO-specific
    reward shaping.
    """

    def __init__(self, instance, config: PPOConfig):
        self.config = config
        self.env = RollingSchedulingEnv(instance)
        self.max_jobs = config.limits["max_jobs"]
        self.max_machines = config.limits["max_machines"]
        self.max_split = config.limits["max_split"]
        self.job_feature_dim = 12
        self.machine_feature_dim = 8
        self.global_feature_dim = 5
        self.obs_dim = (
            self.max_jobs * self.job_feature_dim
            + self.max_machines * self.machine_feature_dim
            + self.global_feature_dim
        )
        self.action_dim = self.max_jobs * self.max_split
        self.illegal_action_count = 0
        self.legal_action_count = 0
        self.selected_split_nums: List[int] = []
        self.reference_cmax = 1.0
        self.reset(instance)

    def reset(self, instance=None) -> Tuple[np.ndarray, np.ndarray]:
        self.env.reset(instance)
        self.jobs = sorted(self.env.instance.jobs, key=lambda j: j.job_id)
        self.machines = sorted(self.env.instance.machines, key=lambda m: m.machine_id)
        self.job_id_to_slot = {job.job_id: idx for idx, job in enumerate(self.jobs)}
        self.illegal_action_count = 0
        self.legal_action_count = 0
        self.selected_split_nums = []
        self.reference_cmax = self._estimate_reference_cmax()
        return self.get_observation(), self.get_action_mask()

    def step(self, action_id: int):
        """Flattened-action compatibility path."""

        mask = self.get_action_mask()
        if action_id < 0 or action_id >= self.action_dim or not mask[action_id]:
            return self._illegal_step(mask)

        job_id, split_num = self.decode_action(action_id)
        return self._step_job_split(job_id, split_num)

    def step_two_head(self, job_slot: int, split_index: int):
        job_mask = self.get_job_mask()
        split_masks = self.get_split_masks()
        if (
            job_slot < 0
            or job_slot >= self.max_jobs
            or split_index < 0
            or split_index >= self.max_split
            or not job_mask[job_slot]
            or not split_masks[job_slot, split_index]
        ):
            return self._illegal_step(self.get_action_mask())

        job_id = self.jobs[job_slot].job_id
        split_num = split_index + 1
        return self._step_job_split(job_id, split_num)

    def step_order_only(self, job_slot: int):
        job_mask = self.get_job_mask()
        if job_slot < 0 or job_slot >= self.max_jobs or not job_mask[job_slot]:
            return self._illegal_step(self.get_action_mask())

        job_id = self.jobs[job_slot].job_id
        split_num = self.choose_split_num(job_id)
        return self._step_job_split(job_id, split_num)

    def decode_action(self, action_id: int) -> Tuple[str, int]:
        job_slot = action_id // self.max_split
        split_num = action_id % self.max_split + 1
        return self.jobs[job_slot].job_id, split_num

    def action_id(self, job_id: str, split_num: int) -> int:
        return self.job_id_to_slot[job_id] * self.max_split + (split_num - 1)

    def get_action_mask(self) -> np.ndarray:
        return self.get_split_masks().reshape(-1)

    def get_job_mask(self) -> np.ndarray:
        mask = np.zeros(self.max_jobs, dtype=np.bool_)
        schedulable = set(self.env.get_schedulable_jobs())
        for slot, job in enumerate(self.jobs[: self.max_jobs]):
            mask[slot] = job.job_id in schedulable
        return mask

    def get_split_masks(self) -> np.ndarray:
        masks = np.zeros((self.max_jobs, self.max_split), dtype=np.bool_)
        schedulable = set(self.env.get_schedulable_jobs())
        for slot, job in enumerate(self.jobs[: self.max_jobs]):
            if job.job_id not in schedulable:
                continue
            max_legal_split = min(job.max_split_num, len(job.candidate_machines), self.max_split)
            masks[slot, :max_legal_split] = True
        return masks

    def get_policy_masks(self) -> Dict[str, np.ndarray]:
        return {
            "flat": self.get_action_mask(),
            "job": self.get_job_mask(),
            "split": self.get_split_masks(),
        }

    def get_observation(self) -> np.ndarray:
        scale = self._time_scale()
        decision_time = self.env._current_decision_time()
        current_period_idx = int(decision_time // self.env.instance.rolling_period_length)

        job_features = np.zeros((self.max_jobs, self.job_feature_dim), dtype=np.float32)
        for slot, job in enumerate(self.jobs[: self.max_jobs]):
            times = list(self.env.instance.processing_time[job.job_id].values())
            one_hot = self._process_one_hot(job.process_type)
            is_completed = 1.0 if job.job_id in self.env.scheduled_jobs else 0.0
            is_released = 1.0 if job.release_time <= decision_time else 0.0
            job_features[slot] = np.array(
                one_hot
                + [
                    job.release_time / scale,
                    min(times) / scale,
                    (sum(times) / len(times)) / scale,
                    len(job.candidate_machines) / max(1, self.max_machines),
                    job.max_split_num / max(1, self.max_split),
                    is_released,
                    is_completed,
                    0.0,
                    1.0,
                ],
                dtype=np.float32,
            )

        machine_features = np.zeros((self.max_machines, self.machine_feature_dim), dtype=np.float32)
        busy_by_machine = Counter()
        for subtask in self.env.subtasks:
            busy_by_machine[subtask.machine_id] += subtask.duration
        cmax = max(self.env.current_cmax, 1e-6)
        for slot, machine in enumerate(self.machines[: self.max_machines]):
            one_hot = self._process_one_hot(machine.process_type)
            busy = busy_by_machine[machine.machine_id]
            available = self.env.machine_available_time[machine.machine_id]
            machine_features[slot] = np.array(
                one_hot
                + [
                    available / scale,
                    busy / scale,
                    min(1.0, busy / cmax),
                    1.0 if available > decision_time else 0.0,
                    machine.speed_factor / 1.5,
                ],
                dtype=np.float32,
            )

        global_features = np.array(
            [
                decision_time / scale,
                self.env.current_cmax / scale,
                len(self.env.unscheduled_jobs) / max(1, self.max_jobs),
                current_period_idx / max(1, self.env.instance.num_periods),
                self.config.limits["size_id"] / 2.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([job_features.ravel(), machine_features.ravel(), global_features])

    def observation_stats(self) -> Dict[str, float]:
        obs = self.get_observation()
        return {
            "obs_mean": float(obs.mean()),
            "obs_std": float(obs.std()),
            "obs_min": float(obs.min()),
            "obs_max": float(obs.max()),
        }

    def split_distribution(self) -> Dict[int, int]:
        counts = Counter(self.selected_split_nums)
        return {split_num: counts.get(split_num, 0) for split_num in range(1, 5)}

    def choose_split_num(self, job_id: str) -> int:
        job = self.env.job_by_id[job_id]
        max_legal = min(job.max_split_num, len(job.candidate_machines), self.max_split)
        if self.config.split_rule == "min1":
            return 1
        if self.config.split_rule == "max_feasible":
            return max(1, max_legal)
        return self._greedy_ect_split(job_id, max_legal)

    def _greedy_ect_split(self, job_id: str, max_legal: int) -> int:
        best_split = 1
        best_completion = float("inf")
        for split_num in range(1, max(1, max_legal) + 1):
            trial = self.env.clone()
            try:
                _, _, _, info = trial.step((job_id, split_num))
            except ValueError:
                continue
            completion = float(info["job_completion_time"])
            if completion < best_completion:
                best_completion = completion
                best_split = split_num
        return best_split

    def _step_job_split(self, job_id: str, split_num: int):
        self.legal_action_count += 1
        self.selected_split_nums.append(split_num)
        _, raw_reward, done, info = self.env.step((job_id, split_num))
        scaled_reward = self._shape_reward(raw_reward, done, info)
        next_mask = self.get_action_mask()
        relative_improvement = (
            (self.reference_cmax - self.env.current_cmax) / self.reference_cmax
            if done
            else 0.0
        )
        info.update(
            {
                "illegal_action": False,
                "illegal_action_count": self.illegal_action_count,
                "legal_action_count": self.legal_action_count,
                "action_mask_ratio": float(next_mask.mean()),
                "selected_split_num": split_num,
                "raw_reward": raw_reward,
                "scaled_reward": scaled_reward,
                "reference_Cmax": self.reference_cmax,
                "relative_improvement_vs_reference": relative_improvement,
                "selected_job": job_id,
            }
        )
        return self.get_observation(), scaled_reward, done, info, next_mask

    def _illegal_step(self, mask: np.ndarray):
        self.illegal_action_count += 1
        done = self.env.is_done()
        penalty = self.config.illegal_action_penalty * self.config.reward_scale
        return self.get_observation(), penalty, done, {
            "illegal_action": True,
            "illegal_action_count": self.illegal_action_count,
            "legal_action_count": self.legal_action_count,
            "action_mask_ratio": float(mask.mean()) if mask.size else 0.0,
            "raw_reward": 0.0,
            "scaled_reward": penalty,
            "reference_Cmax": self.reference_cmax,
            "relative_improvement_vs_reference": 0.0,
            "selected_job": "",
            "selected_split_num": 0,
        }, mask

    def _shape_reward(self, raw_reward: float, done: bool, info: Dict) -> float:
        if self.config.reward_mode != "normalized_delta_plus_baseline_final":
            reward = raw_reward * self.config.reward_scale
            if done:
                final_cmax = info["final_makespan"]
                final_reward = self.config.final_reward_beta * (self.reference_cmax - final_cmax) / self.reference_cmax
                if self.config.reward_clip is not None:
                    final_reward = float(np.clip(final_reward, -self.config.reward_clip, self.config.reward_clip))
                reward += final_reward
            return reward

        reward = raw_reward / self.reference_cmax if self.config.use_reward_normalization else raw_reward
        if self.config.reward_clip is not None:
            reward = float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))
        if done:
            final_cmax = info["final_makespan"]
            if self.config.use_reward_normalization:
                final_reward = self.config.final_reward_beta * (self.reference_cmax - final_cmax) / self.reference_cmax
                if self.config.reward_clip is not None:
                    final_reward = float(np.clip(final_reward, -self.config.reward_clip, self.config.reward_clip))
                reward += final_reward
            else:
                final_reward = self.config.final_reward_beta * (self.reference_cmax - final_cmax) / self.reference_cmax
                if self.config.reward_clip is not None:
                    final_reward = float(np.clip(final_reward, -self.config.reward_clip, self.config.reward_clip))
                reward += final_reward
        return reward * self.config.reward_scale

    def _estimate_reference_cmax(self) -> float:
        key = (self.env.instance.name, self.config.reference_baseline)
        if key in _REFERENCE_CMAX_CACHE:
            return _REFERENCE_CMAX_CACHE[key]

        reference = 0.0
        if self.config.reference_baseline:
            try:
                reference = run_heuristic(self.env.instance, self.config.reference_baseline).metrics["Cmax_roll"]
            except Exception:
                reference = 0.0
        if reference <= 1e-6:
            reference = self._fallback_reference_cmax()
        reference = max(reference, 1e-6)
        _REFERENCE_CMAX_CACHE[key] = reference
        return reference

    def _fallback_reference_cmax(self) -> float:
        total_min_time = 0.0
        for job in self.env.instance.jobs:
            total_min_time += min(self.env.instance.processing_time[job.job_id].values())
        return max(total_min_time / max(1, len(self.env.instance.machines)), self._time_scale())

    def _process_one_hot(self, process_type: str) -> List[float]:
        return [1.0 if process_type == value else 0.0 for value in PROCESS_TYPES]

    def _time_scale(self) -> float:
        horizon = self.env.instance.rolling_period_length * self.env.instance.num_periods
        return max(1.0, horizon, self.reference_cmax)
