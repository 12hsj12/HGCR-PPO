"""Vector observation encoder and flattened action wrapper."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from configs.ppo_config import PPOConfig
from src.envs.rolling_scheduling_env import RollingSchedulingEnv


PROCESS_TYPES = ["sl", "cu", "co"]


class VectorSchedulingWrapper:
    """Padded vector-state interface for PPO.

    The underlying scheduling environment remains responsible for feasibility,
    ECT machine selection, split ratios, and time updates.
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
        self.reset(instance)

    def reset(self, instance=None) -> Tuple[np.ndarray, np.ndarray]:
        self.env.reset(instance)
        self.jobs = sorted(self.env.instance.jobs, key=lambda j: j.job_id)
        self.machines = sorted(self.env.instance.machines, key=lambda m: m.machine_id)
        self.job_id_to_slot = {job.job_id: idx for idx, job in enumerate(self.jobs)}
        self.illegal_action_count = 0
        self.legal_action_count = 0
        self.selected_split_nums = []
        return self.get_observation(), self.get_action_mask()

    def step(self, action_id: int):
        mask = self.get_action_mask()
        if action_id < 0 or action_id >= self.action_dim or not mask[action_id]:
            self.illegal_action_count += 1
            done = self.env.is_done()
            return self.get_observation(), self.config.illegal_action_penalty, done, {
                "illegal_action": True,
                "illegal_action_count": self.illegal_action_count,
                "action_mask_ratio": float(mask.mean()),
            }, mask

        job_id, split_num = self.decode_action(action_id)
        self.legal_action_count += 1
        self.selected_split_nums.append(split_num)
        obs, reward, done, info = self.env.step((job_id, split_num))
        if done:
            reward += -self.config.alpha_final_reward * info["final_makespan"]
        next_mask = self.get_action_mask()
        info.update(
            {
                "illegal_action": False,
                "illegal_action_count": self.illegal_action_count,
                "legal_action_count": self.legal_action_count,
                "action_mask_ratio": float(next_mask.mean()),
                "selected_split_num": split_num,
            }
        )
        return self.get_observation(), reward, done, info, next_mask

    def decode_action(self, action_id: int) -> Tuple[str, int]:
        job_slot = action_id // self.max_split
        split_num = action_id % self.max_split + 1
        return self.jobs[job_slot].job_id, split_num

    def action_id(self, job_id: str, split_num: int) -> int:
        return self.job_id_to_slot[job_id] * self.max_split + (split_num - 1)

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=np.bool_)
        schedulable = set(self.env.get_schedulable_jobs())
        for slot, job in enumerate(self.jobs[: self.max_jobs]):
            if job.job_id not in schedulable:
                continue
            max_legal_split = min(job.max_split_num, len(job.candidate_machines), self.max_split)
            for split_num in range(1, max_legal_split + 1):
                mask[slot * self.max_split + (split_num - 1)] = True
        return mask

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

    def _process_one_hot(self, process_type: str) -> List[float]:
        return [1.0 if process_type == value else 0.0 for value in PROCESS_TYPES]

    def _time_scale(self) -> float:
        return max(1.0, self.env.instance.rolling_period_length * self.env.instance.num_periods)

    def split_distribution(self) -> Dict[int, int]:
        counts = Counter(self.selected_split_nums)
        return {split_num: counts.get(split_num, 0) for split_num in range(1, self.max_split + 1)}
