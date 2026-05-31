"""Shared Stage C utilities for HybridTopK candidate ranking."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from candidate_generator import (
    fifo_ranked,
    generate_candidates,
    greedy_ect_ranked,
    lookahead_ranked,
    minload_ranked,
    spt_ranked,
)
from src.baselines.heuristics import (
    candidate_load,
    choose_split_num,
    estimated_completion_time,
    lookahead_score,
    mean_candidate_processing_time,
)


FEATURE_NAMES = [
    "release_time_norm",
    "min_processing_time",
    "mean_processing_time",
    "max_processing_time",
    "candidate_machine_count",
    "max_split_num",
    "current_waiting_time",
    "fifo_rank_norm",
    "spt_rank_norm",
    "greedy_ect_score_norm",
    "lookahead_score_norm",
    "minload_score_norm",
    "estimated_completion_time",
    "estimated_waiting_time",
    "current_candidate_machine_load_mean",
    "current_candidate_machine_load_min",
    "current_candidate_machine_load_max",
]


def process_types_for_instance(instance) -> List[str]:
    return list(getattr(instance, "process_types", sorted({job.process_type for job in instance.jobs})))


def feature_dim(instance) -> int:
    return len(FEATURE_NAMES) + len(process_types_for_instance(instance))


def current_time(env) -> float:
    return float(env._current_decision_time())


def rollout_cmax(env, first_job_id: str, rollout_policy: str = "fifo") -> float:
    """Evaluate a first action by completing the episode with a fixed policy."""

    trial = env.clone()
    trial.step((first_job_id, choose_split_num(trial, first_job_id)))
    while not trial.is_done():
        jobs = trial.get_schedulable_jobs()
        if rollout_policy == "lookahead":
            job_id = min(jobs, key=lambda j: (lookahead_score(trial, j), j))
        else:
            job_id = min(jobs, key=lambda j: (trial.job_by_id[j].release_time, j))
        trial.step((job_id, choose_split_num(trial, job_id)))
    return float(trial.current_cmax)


def oracle_cmax_per_candidate(env, candidates: Sequence[str], rollout_policy: str = "fifo") -> List[float]:
    return [rollout_cmax(env, job_id, rollout_policy=rollout_policy) for job_id in candidates]


def best_candidate_index(cmax_values: Sequence[float]) -> int:
    return min(range(len(cmax_values)), key=lambda idx: (float(cmax_values[idx]), idx))


def _rank_map(order: Sequence[str]) -> Dict[str, int]:
    return {job_id: idx for idx, job_id in enumerate(order)}


def _norm_score(value: float, values: Sequence[float]) -> float:
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return 0.0
    return (float(value) - lo) / (hi - lo)


def extract_candidate_features(env, candidates: Sequence[str]) -> List[List[float]]:
    """Return per-candidate numeric features for the current decision state."""

    schedulable = env.get_schedulable_jobs()
    process_types = process_types_for_instance(env.instance)
    process_index = {process_type: idx for idx, process_type in enumerate(process_types)}
    now = current_time(env)
    max_release = max((job.release_time for job in env.instance.jobs), default=1.0) or 1.0
    rank_denominator = max(1, len(schedulable) - 1)

    fifo_ranks = _rank_map(fifo_ranked(env))
    spt_ranks = _rank_map(spt_ranked(env))
    greedy_scores = {job_id: estimated_completion_time(env, job_id) for job_id in schedulable}
    lookahead_scores = {job_id: lookahead_score(env, job_id) for job_id in schedulable}
    minload_scores = {job_id: candidate_load(env, job_id) for job_id in schedulable}

    greedy_values = list(greedy_scores.values()) or [0.0]
    lookahead_values = list(lookahead_scores.values()) or [0.0]
    minload_values = list(minload_scores.values()) or [0.0]

    rows: List[List[float]] = []
    for job_id in candidates:
        job = env.job_by_id[job_id]
        processing = [env.instance.processing_time[job_id][m] for m in job.candidate_machines]
        loads = [env.machine_available_time[m] for m in job.candidate_machines]
        base = [
            float(job.release_time) / max_release,
            min(processing),
            mean_candidate_processing_time(env, job_id),
            max(processing),
            float(len(job.candidate_machines)),
            float(job.max_split_num),
            max(0.0, now - job.release_time),
            float(fifo_ranks.get(job_id, rank_denominator)) / rank_denominator,
            float(spt_ranks.get(job_id, rank_denominator)) / rank_denominator,
            _norm_score(greedy_scores.get(job_id, 0.0), greedy_values),
            _norm_score(lookahead_scores.get(job_id, 0.0), lookahead_values),
            _norm_score(minload_scores.get(job_id, 0.0), minload_values),
            estimated_completion_time(env, job_id),
            max(0.0, now - job.release_time),
            sum(loads) / len(loads),
            min(loads),
            max(loads),
        ]
        one_hot = [0.0] * len(process_types)
        one_hot[process_index[job.process_type]] = 1.0
        rows.append(base + one_hot)
    return rows


def hybrid_candidates(env, top_k: int) -> List[str]:
    return generate_candidates(env, candidate_mode="hybrid_topk", top_k=top_k, fallback_to_all=True)


def fifo_first(env) -> str | None:
    ranked = fifo_ranked(env)
    return ranked[0] if ranked else None


def load_ranker_records(path: Path) -> List[Dict]:
    with path.open("rb") as f:
        return pickle.load(f)


def save_ranker_records(records: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(list(records), f)


def dataset_path(output_dir: str | Path, size: str, split: str, top_k: int) -> Path:
    return Path(output_dir) / f"ranker_dataset_{size}_{split}_topk{int(top_k)}.pkl"


def as_float_array(features: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(features, dtype=np.float32)
