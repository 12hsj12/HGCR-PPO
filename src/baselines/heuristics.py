"""Dispatching-rule baselines using the environment step interface."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from src.envs.rolling_scheduling_env import RollingSchedulingEnv
from src.evaluation.metrics import compute_metrics


@dataclass
class HeuristicResult:
    name: str
    metrics: Dict[str, float]
    schedule: Dict
    env: RollingSchedulingEnv


def _mean_candidate_processing_time(env: RollingSchedulingEnv, job_id: str) -> float:
    times = [env.instance.processing_time[job_id][m] for m in env.job_by_id[job_id].candidate_machines]
    return sum(times) / len(times)


def _choose_split_num(env: RollingSchedulingEnv, job_id: str) -> int:
    job = env.job_by_id[job_id]
    return min(job.max_split_num, len(job.candidate_machines))


def _candidate_load(env: RollingSchedulingEnv, job_id: str) -> float:
    candidates = env.job_by_id[job_id].candidate_machines
    return sum(env.machine_available_time[m] for m in candidates) / len(candidates)


def _greedy_completion(env: RollingSchedulingEnv, job_id: str) -> float:
    split_num = _choose_split_num(env, job_id)
    trial = env.clone()
    _, _, _, info = trial.step((job_id, split_num))
    return info["job_completion_time"]


POLICIES: Dict[str, Callable[[RollingSchedulingEnv, random.Random], str]] = {
    "Random": lambda env, rng: rng.choice(env.get_schedulable_jobs()),
    "FIFO": lambda env, rng: min(env.get_schedulable_jobs(), key=lambda j: (env.job_by_id[j].release_time, j)),
    "SPT": lambda env, rng: min(env.get_schedulable_jobs(), key=lambda j: (_mean_candidate_processing_time(env, j), j)),
    "LPT": lambda env, rng: max(env.get_schedulable_jobs(), key=lambda j: (_mean_candidate_processing_time(env, j), j)),
    "MinCandidateLoad": lambda env, rng: min(env.get_schedulable_jobs(), key=lambda j: (_candidate_load(env, j), env.job_by_id[j].release_time, j)),
    "GreedyECT": lambda env, rng: min(env.get_schedulable_jobs(), key=lambda j: (_greedy_completion(env, j), j)),
}


def run_heuristic(
    instance,
    heuristic_name: str,
    seed: int = 42,
    env: Optional[RollingSchedulingEnv] = None,
) -> HeuristicResult:
    if heuristic_name not in POLICIES:
        raise ValueError(f"Unknown heuristic {heuristic_name!r}. Expected one of {sorted(POLICIES)}.")

    rng = random.Random(seed)
    env = env or RollingSchedulingEnv(instance)
    env.reset(instance)
    policy = POLICIES[heuristic_name]

    while not env.is_done():
        schedulable_jobs = env.get_schedulable_jobs()
        if not schedulable_jobs:
            raise RuntimeError("No schedulable job found although episode is not done.")
        job_id = policy(env, rng)
        env.step((job_id, _choose_split_num(env, job_id)))

    return HeuristicResult(
        name=heuristic_name,
        metrics=compute_metrics(env),
        schedule={"jobs": env.scheduled_jobs, "subtasks": env.subtasks},
        env=env,
    )


def run_all_heuristics(instance, seed: int = 42) -> List[HeuristicResult]:
    return [run_heuristic(instance, name, seed=seed) for name in POLICIES]

