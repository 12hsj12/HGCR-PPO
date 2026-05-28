"""RL-compatible rolling scheduling environment."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from src.core import JobSchedule, SchedulingInstance, SubTask
from src.instances.instance_generator import generate_instance


Action = Tuple[str, int]


class RollingSchedulingEnv:
    """Environment using action ``(job_id, split_num)``.

    Rolling periods are represented by release times and period-start lower
    bounds. Already scheduled subtasks are never changed; machine availability
    carries all cross-period commitments forward.
    """

    def __init__(self, instance: Optional[SchedulingInstance] = None):
        self.instance = instance
        self.reset(instance)

    def reset(self, instance: Optional[SchedulingInstance] = None) -> Dict:
        if instance is not None:
            self.instance = instance
        if self.instance is None:
            self.instance = generate_instance("small", seed=42)

        self.job_by_id = self.instance.job_by_id
        self.machine_by_id = self.instance.machine_by_id
        self.machine_available_time: Dict[str, float] = {m.machine_id: 0.0 for m in self.instance.machines}
        self.unscheduled_jobs = {job.job_id for job in self.instance.jobs}
        self.scheduled_jobs: Dict[str, JobSchedule] = {}
        self.subtasks: List[SubTask] = []
        self.current_cmax = 0.0
        return self.get_state()

    def get_state(self) -> Dict:
        next_release = min((self.job_by_id[j].release_time for j in self.unscheduled_jobs), default=None)
        active_jobs = self.get_schedulable_jobs()
        return {
            "current_cmax": self.current_cmax,
            "machine_available_time": deepcopy(self.machine_available_time),
            "unscheduled_jobs": sorted(self.unscheduled_jobs),
            "active_jobs": active_jobs,
            "next_release_time": next_release,
            "num_scheduled_jobs": len(self.scheduled_jobs),
        }

    def is_done(self) -> bool:
        return not self.unscheduled_jobs

    def get_schedulable_jobs(self) -> List[str]:
        if not self.unscheduled_jobs:
            return []
        frontier = self._current_decision_time()
        return sorted(j for j in self.unscheduled_jobs if self.job_by_id[j].release_time <= frontier)

    def step(self, action: Action):
        if self.is_done():
            return self.get_state(), 0.0, True, {"final_makespan": self.current_cmax}

        job_id, split_num = action
        if job_id not in self.unscheduled_jobs:
            raise ValueError(f"Job {job_id!r} is not unscheduled.")

        job = self.job_by_id[job_id]
        earliest_decision_time = self._current_decision_time()
        if job.release_time > earliest_decision_time:
            raise ValueError(
                f"Job {job_id!r} is not released at current decision time "
                f"{earliest_decision_time:.3f}; release_time={job.release_time:.3f}."
            )

        selected_machines = self._select_machines_by_ect(job_id, int(split_num))
        ratios = self._compute_split_ratios(job_id, selected_machines)
        old_cmax = self.current_cmax
        period_start = self._period_start(job.release_time)

        subtasks: List[SubTask] = []
        for machine_id in selected_machines:
            start_time = max(self.machine_available_time[machine_id], job.release_time, period_start)
            duration = ratios[machine_id] * self.instance.processing_time[job_id][machine_id]
            completion_time = start_time + duration
            subtask = SubTask(
                job_id=job_id,
                machine_id=machine_id,
                process_type=job.process_type,
                ratio=ratios[machine_id],
                start_time=start_time,
                duration=duration,
                completion_time=completion_time,
            )
            subtasks.append(subtask)
            self.machine_available_time[machine_id] = completion_time

        completion_time = max(subtask.completion_time for subtask in subtasks)
        start_time = min(subtask.start_time for subtask in subtasks)
        self.scheduled_jobs[job_id] = JobSchedule(
            job_id=job_id,
            process_type=job.process_type,
            release_time=job.release_time,
            start_time=start_time,
            completion_time=completion_time,
            subtasks=subtasks,
        )
        self.subtasks.extend(subtasks)
        self.unscheduled_jobs.remove(job_id)
        self.current_cmax = max(self.current_cmax, completion_time)

        reward = -(self.current_cmax - old_cmax)
        done = self.is_done()
        info = {
            "job_id": job_id,
            "selected_machines": selected_machines,
            "split_ratios": ratios,
            "job_completion_time": completion_time,
            "current_makespan": self.current_cmax,
        }
        if done:
            info["final_makespan"] = self.current_cmax
        return self.get_state(), reward, done, info

    def render_gantt(self, save_path: str = "data/results/gantt.png") -> str:
        from src.visualization import plot_gantt

        return plot_gantt(self, save_path=save_path)

    def clone(self) -> "RollingSchedulingEnv":
        return deepcopy(self)

    def _current_decision_time(self) -> float:
        active_releases = [self.job_by_id[j].release_time for j in self.unscheduled_jobs if self.job_by_id[j].release_time <= self.current_cmax]
        if active_releases:
            return self.current_cmax
        return min((self.job_by_id[j].release_time for j in self.unscheduled_jobs), default=self.current_cmax)

    def _period_start(self, time_value: float) -> float:
        period_idx = int(time_value // self.instance.rolling_period_length)
        return period_idx * self.instance.rolling_period_length

    def _select_machines_by_ect(self, job_id: str, split_num: int) -> List[str]:
        job = self.job_by_id[job_id]
        if split_num < 1:
            raise ValueError("split_num must be at least 1.")
        split_num = min(split_num, job.max_split_num, len(job.candidate_machines))
        if split_num < 1:
            raise ValueError(f"Job {job_id!r} has no eligible machines.")

        period_start = self._period_start(job.release_time)
        ect_values = []
        for machine_id in job.candidate_machines:
            machine = self.machine_by_id[machine_id]
            if machine.process_type != job.process_type:
                raise ValueError(f"Candidate machine {machine_id!r} has wrong process type.")
            full_time = self.instance.processing_time[job_id][machine_id]
            start = max(self.machine_available_time[machine_id], job.release_time, period_start)
            ect_values.append((start + full_time, machine_id))
        ect_values.sort(key=lambda item: (item[0], item[1]))
        return [machine_id for _, machine_id in ect_values[:split_num]]

    def _compute_split_ratios(self, job_id: str, selected_machines: List[str]) -> Dict[str, float]:
        inverse_times = {m: 1.0 / self.instance.processing_time[job_id][m] for m in selected_machines}
        denominator = sum(inverse_times.values())
        return {machine_id: inverse_time / denominator for machine_id, inverse_time in inverse_times.items()}

