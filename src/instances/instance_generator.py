"""Reproducible instance generator for rolling steel-coil scheduling."""

from __future__ import annotations

import random
from typing import Dict, List

from src.core import Job, Machine, SchedulingInstance


SIZE_CONFIGS = {
    "small": {"jobs": 24, "machines": {"sl": 3, "cu": 2, "co": 2}, "periods": 4, "period_len": 40.0},
    "medium": {"jobs": 64, "machines": {"sl": 5, "cu": 4, "co": 3}, "periods": 6, "period_len": 45.0},
    "large": {"jobs": 132, "machines": {"sl": 7, "cu": 6, "co": 5}, "periods": 10, "period_len": 50.0},
}

BASE_PROCESS_TIMES = {"sl": (18.0, 60.0), "cu": (14.0, 52.0), "co": (22.0, 72.0)}
PROCESS_WEIGHTS = [("sl", 0.42), ("cu", 0.34), ("co", 0.24)]


def _weighted_process_type(rng: random.Random) -> str:
    draw = rng.random()
    cumulative = 0.0
    for process_type, weight in PROCESS_WEIGHTS:
        cumulative += weight
        if draw <= cumulative:
            return process_type
    return PROCESS_WEIGHTS[-1][0]


def generate_instance(size: str = "small", seed: int = 42) -> SchedulingInstance:
    """Generate a deterministic scheduling instance.

    Machines are unrelated/non-identical: even within the same process type,
    each line receives a different speed factor and each job has a small
    machine-specific processing perturbation.
    """

    if size not in SIZE_CONFIGS:
        raise ValueError(f"Unknown size {size!r}. Expected one of {sorted(SIZE_CONFIGS)}.")

    rng = random.Random(seed)
    config = SIZE_CONFIGS[size]
    rolling_period_length = config["period_len"]
    num_periods = config["periods"]

    machines: List[Machine] = []
    machines_by_process: Dict[str, List[str]] = {}
    for process_type, count in config["machines"].items():
        machines_by_process[process_type] = []
        for idx in range(count):
            machine_id = f"{process_type}_m{idx + 1}"
            # Spread speeds enough to make line heterogeneity visible.
            speed_factor = round(rng.uniform(0.75, 1.35), 3)
            machines.append(Machine(machine_id=machine_id, process_type=process_type, speed_factor=speed_factor))
            machines_by_process[process_type].append(machine_id)

    jobs: List[Job] = []
    processing_time: Dict[str, Dict[str, float]] = {}
    horizon_release_end = rolling_period_length * max(1, num_periods - 1)

    for idx in range(config["jobs"]):
        process_type = _weighted_process_type(rng)
        candidate_pool = machines_by_process[process_type]
        min_candidates = min(2, len(candidate_pool))
        candidate_count = rng.randint(min_candidates, len(candidate_pool))
        candidate_machines = sorted(rng.sample(candidate_pool, candidate_count))
        max_split_num = rng.randint(1, min(3, len(candidate_machines)))
        period_idx = rng.randrange(num_periods)
        release_floor = period_idx * rolling_period_length
        release_ceiling = min(release_floor + rolling_period_length * 0.8, horizon_release_end)
        release_time = round(rng.uniform(release_floor, release_ceiling), 2)
        job_id = f"j{idx + 1:03d}"

        jobs.append(
            Job(
                job_id=job_id,
                process_type=process_type,
                release_time=release_time,
                candidate_machines=candidate_machines,
                max_split_num=max_split_num,
            )
        )

        low, high = BASE_PROCESS_TIMES[process_type]
        base_time = rng.uniform(low, high)
        processing_time[job_id] = {}
        for machine in machines:
            if machine.machine_id in candidate_machines:
                perturbation = rng.uniform(0.88, 1.16)
                processing_time[job_id][machine.machine_id] = round(base_time * perturbation / machine.speed_factor, 3)

    jobs.sort(key=lambda job: (job.release_time, job.job_id))
    return SchedulingInstance(
        jobs=jobs,
        machines=machines,
        processing_time=processing_time,
        rolling_period_length=rolling_period_length,
        num_periods=num_periods,
        name=f"{size}_seed_{seed}",
    )

