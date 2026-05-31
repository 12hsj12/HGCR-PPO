"""Candidate-set generation utilities for HGCR-PPO Stage B."""

from __future__ import annotations

from typing import Callable, Dict, List

from src.baselines.heuristics import (
    candidate_load,
    estimated_completion_time,
    lookahead_score,
    mean_candidate_processing_time,
)


CANDIDATE_MODES = [
    "all",
    "fifo_topk",
    "spt_topk",
    "greedy_ect_topk",
    "minload_topk",
    "lookahead_topk",
    "hybrid_topk",
    "beam_topk",
]


def _dedupe(items: List[str], allow_duplicate: bool) -> List[str]:
    if allow_duplicate:
        return items
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _ranked(env, key_fn: Callable[[str], tuple], top_k: int, allow_duplicate: bool) -> List[str]:
    jobs = sorted(env.get_schedulable_jobs(), key=key_fn)
    return _dedupe(jobs[:top_k], allow_duplicate)


def fifo_ranked(env) -> List[str]:
    return sorted(env.get_schedulable_jobs(), key=lambda j: (env.job_by_id[j].release_time, j))


def spt_ranked(env) -> List[str]:
    return sorted(env.get_schedulable_jobs(), key=lambda j: (mean_candidate_processing_time(env, j), j))


def greedy_ect_ranked(env) -> List[str]:
    return sorted(env.get_schedulable_jobs(), key=lambda j: (estimated_completion_time(env, j), j))


def minload_ranked(env) -> List[str]:
    return sorted(env.get_schedulable_jobs(), key=lambda j: (candidate_load(env, j), env.job_by_id[j].release_time, j))


def lookahead_ranked(env) -> List[str]:
    return sorted(env.get_schedulable_jobs(), key=lambda j: (lookahead_score(env, j), j))


def first_rule_jobs(env) -> Dict[str, str | None]:
    rankings = {
        "fifo": fifo_ranked(env),
        "greedy": greedy_ect_ranked(env),
        "lookahead": lookahead_ranked(env),
        "minload": minload_ranked(env),
    }
    return {name: (jobs[0] if jobs else None) for name, jobs in rankings.items()}


def _fifo(env, top_k: int, allow_duplicate: bool) -> List[str]:
    return _ranked(env, lambda j: (env.job_by_id[j].release_time, j), top_k, allow_duplicate)


def _spt(env, top_k: int, allow_duplicate: bool) -> List[str]:
    return _ranked(env, lambda j: (mean_candidate_processing_time(env, j), j), top_k, allow_duplicate)


def _greedy_ect(env, top_k: int, allow_duplicate: bool) -> List[str]:
    return _ranked(env, lambda j: (estimated_completion_time(env, j), j), top_k, allow_duplicate)


def _minload(env, top_k: int, allow_duplicate: bool) -> List[str]:
    return _ranked(env, lambda j: (candidate_load(env, j), env.job_by_id[j].release_time, j), top_k, allow_duplicate)


def _lookahead(env, top_k: int, allow_duplicate: bool) -> List[str]:
    return _ranked(env, lambda j: (lookahead_score(env, j), j), top_k, allow_duplicate)


def _hybrid(env, top_k: int, allow_duplicate: bool) -> List[str]:
    forced = [
        first
        for first in first_rule_jobs(env).values()
        if first is not None
    ]
    forced = _dedupe(forced, allow_duplicate=False)

    merged: List[str] = list(forced)
    for bucket in [
        _fifo(env, top_k, allow_duplicate=True),
        _spt(env, top_k, allow_duplicate=True),
        _greedy_ect(env, top_k, allow_duplicate=True),
        _minload(env, top_k, allow_duplicate=True),
        _lookahead(env, top_k, allow_duplicate=True),
    ]:
        merged.extend(bucket)
    merged = _dedupe(merged, allow_duplicate)
    if len(merged) > top_k:
        forced_set = set(forced)
        extras = [job_id for job_id in merged if job_id not in forced_set]
        slots = max(0, top_k - len(forced))
        extras = sorted(extras, key=lambda j: (lookahead_score(env, j), estimated_completion_time(env, j), j))[:slots]
        merged = forced + extras
    return merged


def generate_candidates(
    env,
    candidate_mode: str = "hybrid_topk",
    top_k: int = 5,
    allow_duplicate: bool = False,
    fallback_to_all: bool = True,
) -> List[str]:
    if candidate_mode not in CANDIDATE_MODES:
        raise ValueError(f"Unknown candidate_mode {candidate_mode!r}. Expected one of {CANDIDATE_MODES}.")

    top_k = max(1, int(top_k))
    all_jobs = env.get_schedulable_jobs()
    if candidate_mode == "all":
        candidates = list(all_jobs)
    elif candidate_mode == "fifo_topk":
        candidates = _fifo(env, top_k, allow_duplicate)
    elif candidate_mode == "spt_topk":
        candidates = _spt(env, top_k, allow_duplicate)
    elif candidate_mode == "greedy_ect_topk":
        candidates = _greedy_ect(env, top_k, allow_duplicate)
    elif candidate_mode == "minload_topk":
        candidates = _minload(env, top_k, allow_duplicate)
    elif candidate_mode == "lookahead_topk":
        candidates = _lookahead(env, top_k, allow_duplicate)
    elif candidate_mode == "beam_topk":
        candidates = _hybrid(env, top_k, allow_duplicate)
    else:
        candidates = _hybrid(env, top_k, allow_duplicate)

    if not candidates and fallback_to_all:
        return list(all_jobs)
    return candidates
