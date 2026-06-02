"""Build Stage D bipartite job-machine graphs for GNN-Ranker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from candidate_generator import (
    fifo_ranked,
    greedy_ect_ranked,
    lookahead_ranked,
    minload_ranked,
    spt_ranked,
)
from src.baselines.heuristics import (
    candidate_load,
    estimated_completion_time,
    lookahead_score,
    mean_candidate_processing_time,
)
from stage_c_utils import current_time, process_types_for_instance


JOB_BASE_FEATURE_NAMES = [
    "release_time_norm",
    "min_processing_time_norm",
    "mean_processing_time_norm",
    "max_processing_time_norm",
    "candidate_machine_count_norm",
    "max_split_num_norm",
    "current_waiting_time_norm",
    "fifo_rank_norm",
    "spt_rank_norm",
    "greedy_ect_score_norm",
    "lookahead_score_norm",
    "minload_score_norm",
    "is_candidate_job",
]
MACHINE_BASE_FEATURE_NAMES = [
    "current_available_time_norm",
    "current_load_norm",
    "machine_utilization",
    "average_processing_time_for_visible_jobs_norm",
    "candidate_job_count_norm",
]
EDGE_FEATURE_NAMES = [
    "processing_time_norm",
    "relative_processing_time",
    "estimated_completion_time_norm",
    "machine_available_time_norm",
    "eligibility_indicator",
]


@dataclass
class BipartiteGraphData:
    """Small dependency-free graph container suitable for PyTorch models."""

    job_ids: List[str]
    machine_ids: List[str]
    candidate_job_ids: List[str]
    job_features: np.ndarray
    machine_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    candidate_job_indices: np.ndarray
    metadata: Dict[str, object]


def job_feature_dim(instance) -> int:
    return len(process_types_for_instance(instance)) + len(JOB_BASE_FEATURE_NAMES)


def machine_feature_dim(instance) -> int:
    return len(process_types_for_instance(instance)) + len(MACHINE_BASE_FEATURE_NAMES)


def edge_feature_dim() -> int:
    return len(EDGE_FEATURE_NAMES)


def build_graph_from_env(env, candidate_job_ids: Sequence[str]) -> BipartiteGraphData:
    """Construct the current task-line bipartite graph from an environment state."""

    visible_jobs = env.get_schedulable_jobs()
    if not visible_jobs:
        raise RuntimeError("Cannot build a GNN graph without schedulable jobs.")

    candidate_job_ids = [job_id for job_id in candidate_job_ids if job_id in visible_jobs]
    if not candidate_job_ids:
        candidate_job_ids = list(visible_jobs)

    process_types = process_types_for_instance(env.instance)
    process_index = {process_type: idx for idx, process_type in enumerate(process_types)}
    time_scale = _time_scale(env)
    proc_scale = _processing_scale(env)
    max_candidates = max(1, max((len(job.candidate_machines) for job in env.instance.jobs), default=1))
    max_split = max(1, max((job.max_split_num for job in env.instance.jobs), default=1))

    job_rows = _job_feature_rows(
        env,
        visible_jobs,
        candidate_job_ids,
        process_types,
        process_index,
        time_scale,
        proc_scale,
        max_candidates,
        max_split,
    )
    machine_rows = _machine_feature_rows(
        env,
        visible_jobs,
        candidate_job_ids,
        process_types,
        process_index,
        time_scale,
        proc_scale,
    )
    edge_index, edge_features = _edge_arrays(env, visible_jobs, process_types, time_scale, proc_scale)

    job_index = {job_id: idx for idx, job_id in enumerate(visible_jobs)}
    candidate_indices = np.asarray([job_index[job_id] for job_id in candidate_job_ids], dtype=np.int64)
    return BipartiteGraphData(
        job_ids=list(visible_jobs),
        machine_ids=[machine.machine_id for machine in env.instance.machines],
        candidate_job_ids=list(candidate_job_ids),
        job_features=np.asarray(job_rows, dtype=np.float32),
        machine_features=np.asarray(machine_rows, dtype=np.float32),
        edge_index=edge_index,
        edge_features=edge_features,
        candidate_job_indices=candidate_indices,
        metadata={
            "current_time": current_time(env),
            "current_cmax": float(env.current_cmax),
            "job_feature_names": [f"process_type={p}" for p in process_types] + JOB_BASE_FEATURE_NAMES,
            "machine_feature_names": [f"process_type={p}" for p in process_types] + MACHINE_BASE_FEATURE_NAMES,
            "edge_feature_names": EDGE_FEATURE_NAMES,
        },
    )


def _job_feature_rows(
    env,
    visible_jobs: Sequence[str],
    candidate_job_ids: Sequence[str],
    process_types: Sequence[str],
    process_index: Dict[str, int],
    time_scale: float,
    proc_scale: float,
    max_candidates: int,
    max_split: int,
) -> List[List[float]]:
    now = current_time(env)
    candidate_set = set(candidate_job_ids)
    rank_denominator = max(1, len(visible_jobs) - 1)
    fifo_ranks = _rank_map(fifo_ranked(env))
    spt_ranks = _rank_map(spt_ranked(env))
    greedy_values = {job_id: estimated_completion_time(env, job_id) for job_id in visible_jobs}
    lookahead_values = {job_id: lookahead_score(env, job_id) for job_id in visible_jobs}
    minload_values = {job_id: candidate_load(env, job_id) for job_id in visible_jobs}

    rows: List[List[float]] = []
    for job_id in visible_jobs:
        job = env.job_by_id[job_id]
        processing = [env.instance.processing_time[job_id][m] for m in job.candidate_machines]
        one_hot = [0.0] * len(process_types)
        one_hot[process_index[job.process_type]] = 1.0
        rows.append(
            one_hot
            + [
                float(job.release_time) / time_scale,
                min(processing) / proc_scale,
                mean_candidate_processing_time(env, job_id) / proc_scale,
                max(processing) / proc_scale,
                float(len(job.candidate_machines)) / max_candidates,
                float(job.max_split_num) / max_split,
                max(0.0, now - job.release_time) / time_scale,
                float(fifo_ranks.get(job_id, rank_denominator)) / rank_denominator,
                float(spt_ranks.get(job_id, rank_denominator)) / rank_denominator,
                _norm_score(greedy_values.get(job_id, 0.0), greedy_values.values()),
                _norm_score(lookahead_values.get(job_id, 0.0), lookahead_values.values()),
                _norm_score(minload_values.get(job_id, 0.0), minload_values.values()),
                1.0 if job_id in candidate_set else 0.0,
            ]
        )
    return rows


def _machine_feature_rows(
    env,
    visible_jobs: Sequence[str],
    candidate_job_ids: Sequence[str],
    process_types: Sequence[str],
    process_index: Dict[str, int],
    time_scale: float,
    proc_scale: float,
) -> List[List[float]]:
    now = max(current_time(env), 1.0)
    candidate_set = set(candidate_job_ids)
    rows: List[List[float]] = []
    for machine in env.instance.machines:
        eligible_visible = [
            job_id for job_id in visible_jobs if machine.machine_id in env.job_by_id[job_id].candidate_machines
        ]
        eligible_candidates = [
            job_id for job_id in candidate_set if machine.machine_id in env.job_by_id[job_id].candidate_machines
        ]
        avg_processing = (
            sum(env.instance.processing_time[job_id][machine.machine_id] for job_id in eligible_visible)
            / len(eligible_visible)
            if eligible_visible
            else 0.0
        )
        available = float(env.machine_available_time[machine.machine_id])
        one_hot = [0.0] * len(process_types)
        one_hot[process_index[machine.process_type]] = 1.0
        rows.append(
            one_hot
            + [
                available / time_scale,
                available / time_scale,
                min(1.0, available / now),
                avg_processing / proc_scale,
                float(len(eligible_candidates)) / max(1, len(candidate_set)),
            ]
        )
    return rows


def _edge_arrays(env, visible_jobs: Sequence[str], process_types: Sequence[str], time_scale: float, proc_scale: float):
    del process_types
    machine_index = {machine.machine_id: idx for idx, machine in enumerate(env.instance.machines)}
    edge_pairs: List[List[int]] = []
    rows: List[List[float]] = []
    for job_idx, job_id in enumerate(visible_jobs):
        job = env.job_by_id[job_id]
        processing_values = [env.instance.processing_time[job_id][m] for m in job.candidate_machines]
        mean_processing = sum(processing_values) / len(processing_values)
        period_start = env._period_start(job.release_time)
        for machine_id in job.candidate_machines:
            processing = float(env.instance.processing_time[job_id][machine_id])
            available = float(env.machine_available_time[machine_id])
            start = max(available, job.release_time, period_start)
            edge_pairs.append([job_idx, machine_index[machine_id]])
            rows.append(
                [
                    processing / proc_scale,
                    processing / max(mean_processing, 1e-6),
                    (start + processing) / time_scale,
                    available / time_scale,
                    1.0,
                ]
            )
    if not edge_pairs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, edge_feature_dim()), dtype=np.float32)
    return np.asarray(edge_pairs, dtype=np.int64).T, np.asarray(rows, dtype=np.float32)


def _processing_scale(env) -> float:
    values = [
        float(value)
        for job_times in env.instance.processing_time.values()
        for value in job_times.values()
    ]
    return max(values, default=1.0) or 1.0


def _time_scale(env) -> float:
    max_release = max((float(job.release_time) for job in env.instance.jobs), default=0.0)
    max_available = max((float(v) for v in env.machine_available_time.values()), default=0.0)
    horizon = env.instance.rolling_period_length * env.instance.num_periods
    return max(1.0, max_release, max_available, float(env.current_cmax), float(horizon))


def _rank_map(order: Sequence[str]) -> Dict[str, int]:
    return {job_id: idx for idx, job_id in enumerate(order)}


def _norm_score(value: float, values) -> float:
    values = list(values) or [0.0]
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return 0.0
    return (float(value) - lo) / (hi - lo)
