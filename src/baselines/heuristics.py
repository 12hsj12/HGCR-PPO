"""Dispatching-rule baselines using the environment step interface."""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import pstdev
from typing import Callable, Dict, List, Optional, Tuple

from src.envs.rolling_scheduling_env import RollingSchedulingEnv
from src.evaluation.metrics import compute_metrics


@dataclass
class HeuristicResult:
    name: str
    metrics: Dict[str, float]
    schedule: Dict
    env: RollingSchedulingEnv


def mean_candidate_processing_time(env: RollingSchedulingEnv, job_id: str) -> float:
    times = [env.instance.processing_time[job_id][m] for m in env.job_by_id[job_id].candidate_machines]
    return sum(times) / len(times)


def choose_split_num(env: RollingSchedulingEnv, job_id: str) -> int:
    job = env.job_by_id[job_id]
    return min(job.max_split_num, len(job.candidate_machines))


def candidate_load(env: RollingSchedulingEnv, job_id: str) -> float:
    candidates = env.job_by_id[job_id].candidate_machines
    return sum(env.machine_available_time[m] for m in candidates) / len(candidates)


def estimated_completion_time(env: RollingSchedulingEnv, job_id: str, split_num: Optional[int] = None) -> float:
    split_num = choose_split_num(env, job_id) if split_num is None else split_num
    selected_machines = env._select_machines_by_ect(job_id, split_num)
    ratios = env._compute_split_ratios(job_id, selected_machines)
    job = env.job_by_id[job_id]
    period_start = env._period_start(job.release_time)
    return max(
        max(env.machine_available_time[machine_id], job.release_time, period_start)
        + ratios[machine_id] * env.instance.processing_time[job_id][machine_id]
        for machine_id in selected_machines
    )


def candidate_machine_load_after_assign(env: RollingSchedulingEnv, job_id: str, split_num: Optional[int] = None) -> float:
    split_num = choose_split_num(env, job_id) if split_num is None else split_num
    selected_machines = env._select_machines_by_ect(job_id, split_num)
    ratios = env._compute_split_ratios(job_id, selected_machines)
    job = env.job_by_id[job_id]
    period_start = env._period_start(job.release_time)
    updated_availability = []
    for machine_id in env.job_by_id[job_id].candidate_machines:
        if machine_id in selected_machines:
            start = max(env.machine_available_time[machine_id], job.release_time, period_start)
            updated_availability.append(start + ratios[machine_id] * env.instance.processing_time[job_id][machine_id])
        else:
            updated_availability.append(env.machine_available_time[machine_id])
    return sum(updated_availability) / len(updated_availability)


def process_remaining_pressure(env: RollingSchedulingEnv, process_type: str) -> float:
    remaining = [job_id for job_id in env.unscheduled_jobs if env.job_by_id[job_id].process_type == process_type]
    machines = [m.machine_id for m in env.instance.machines if m.process_type == process_type]
    if not machines:
        return 0.0
    total_work = sum(mean_candidate_processing_time(env, job_id) for job_id in remaining)
    return total_work / len(machines)


def waiting_time(env: RollingSchedulingEnv, job_id: str) -> float:
    return max(0.0, env._current_decision_time() - env.job_by_id[job_id].release_time)


def lookahead_score(
    env: RollingSchedulingEnv,
    job_id: str,
    lambda_load: float = 0.15,
    lambda_pressure: float = 0.02,
    lambda_wait: float = 0.10,
) -> float:
    """One-step lookahead score; lower is better."""

    job = env.job_by_id[job_id]
    return (
        estimated_completion_time(env, job_id)
        + lambda_load * candidate_machine_load_after_assign(env, job_id)
        + lambda_pressure * process_remaining_pressure(env, job.process_type)
        - lambda_wait * waiting_time(env, job_id)
    )


def choose_lookahead_greedy_job(env: RollingSchedulingEnv, rng: random.Random) -> str:
    return min(env.get_schedulable_jobs(), key=lambda j: (lookahead_score(env, j), j))


def _state_score(env: RollingSchedulingEnv, alpha: float = 0.05, beta: float = 0.02) -> float:
    completions = [job.completion_time for job in env.scheduled_jobs.values()]
    avg_completion = sum(completions) / len(completions) if completions else 0.0
    loads = [0.0 for _ in env.instance.machines]
    machine_index = {machine.machine_id: idx for idx, machine in enumerate(env.instance.machines)}
    for subtask in env.subtasks:
        loads[machine_index[subtask.machine_id]] += subtask.duration
    load_balance = pstdev(loads) if len(loads) > 1 else 0.0
    return float(env.current_cmax) + alpha * avg_completion + beta * load_balance


def _candidate_union_for_beam(env: RollingSchedulingEnv, candidate_top_k: int) -> List[str]:
    jobs = env.get_schedulable_jobs()
    buckets = [
        sorted(jobs, key=lambda j: (env.job_by_id[j].release_time, j)),
        sorted(jobs, key=lambda j: (estimated_completion_time(env, j), j)),
        sorted(jobs, key=lambda j: (lookahead_score(env, j), j)),
    ]
    merged: List[str] = []
    for bucket in buckets:
        for job_id in bucket[:candidate_top_k]:
            if job_id not in merged:
                merged.append(job_id)
    return merged[:candidate_top_k]


def choose_beam_search_job(
    env: RollingSchedulingEnv,
    rng: random.Random,
    beam_width: int = 3,
    candidate_top_k: int = 5,
) -> str:
    """Return the first action from a lightweight receding-horizon beam search."""

    beams: List[Tuple[RollingSchedulingEnv, Optional[str]]] = [(env.clone(), None)]
    while beams and not all(beam_env.is_done() for beam_env, _ in beams):
        expanded: List[Tuple[float, RollingSchedulingEnv, str]] = []
        for beam_env, first_job in beams:
            if beam_env.is_done():
                if first_job is not None:
                    expanded.append((_state_score(beam_env), beam_env, first_job))
                continue
            for job_id in _candidate_union_for_beam(beam_env, candidate_top_k):
                trial = beam_env.clone()
                trial.step((job_id, choose_split_num(trial, job_id)))
                expanded.append((_state_score(trial), trial, first_job or job_id))
        if not expanded:
            break
        expanded.sort(key=lambda item: item[0])
        beams = [(beam_env, first_job) for _, beam_env, first_job in expanded[:beam_width]]

    if not beams or beams[0][1] is None:
        return choose_lookahead_greedy_job(env, rng)
    return beams[0][1]


POLICIES: Dict[str, Callable[[RollingSchedulingEnv, random.Random], str]] = {
    "Random": lambda env, rng: rng.choice(env.get_schedulable_jobs()),
    "FIFO": lambda env, rng: min(env.get_schedulable_jobs(), key=lambda j: (env.job_by_id[j].release_time, j)),
    "SPT": lambda env, rng: min(env.get_schedulable_jobs(), key=lambda j: (mean_candidate_processing_time(env, j), j)),
    "LPT": lambda env, rng: max(env.get_schedulable_jobs(), key=lambda j: (mean_candidate_processing_time(env, j), j)),
    "MinCandidateLoad": lambda env, rng: min(env.get_schedulable_jobs(), key=lambda j: (candidate_load(env, j), env.job_by_id[j].release_time, j)),
    "GreedyECT": lambda env, rng: min(env.get_schedulable_jobs(), key=lambda j: (estimated_completion_time(env, j), j)),
}

STRONG_POLICIES: Dict[str, Callable[[RollingSchedulingEnv, random.Random], str]] = {
    "LookaheadGreedy": choose_lookahead_greedy_job,
    "BeamSearch": choose_beam_search_job,
}
ALL_POLICIES = {**POLICIES, **STRONG_POLICIES}


def run_heuristic(
    instance,
    heuristic_name: str,
    seed: int = 42,
    env: Optional[RollingSchedulingEnv] = None,
) -> HeuristicResult:
    if heuristic_name not in ALL_POLICIES:
        raise ValueError(f"Unknown heuristic {heuristic_name!r}. Expected one of {sorted(ALL_POLICIES)}.")

    rng = random.Random(seed)
    env = env or RollingSchedulingEnv(instance)
    env.reset(instance)
    policy = ALL_POLICIES[heuristic_name]

    while not env.is_done():
        schedulable_jobs = env.get_schedulable_jobs()
        if not schedulable_jobs:
            raise RuntimeError("No schedulable job found although episode is not done.")
        job_id = policy(env, rng)
        env.step((job_id, choose_split_num(env, job_id)))

    return HeuristicResult(
        name=heuristic_name,
        metrics=compute_metrics(env),
        schedule={"jobs": env.scheduled_jobs, "subtasks": env.subtasks},
        env=env,
    )


def run_all_heuristics(instance, seed: int = 42) -> List[HeuristicResult]:
    return [run_heuristic(instance, name, seed=seed) for name in POLICIES]
