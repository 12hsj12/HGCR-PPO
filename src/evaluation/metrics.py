"""Metrics for rolling scheduling schedules."""

from __future__ import annotations

from statistics import pstdev
from typing import Dict


def compute_metrics(env) -> Dict[str, float]:
    jobs = list(env.scheduled_jobs.values())
    subtasks = list(env.subtasks)
    cmax = max((job.completion_time for job in jobs), default=0.0)
    avg_completion = sum(job.completion_time for job in jobs) / len(jobs) if jobs else 0.0
    avg_waiting = sum(job.start_time - job.release_time for job in jobs) / len(jobs) if jobs else 0.0

    machine_busy = {machine.machine_id: 0.0 for machine in env.instance.machines}
    for subtask in subtasks:
        machine_busy[subtask.machine_id] += subtask.duration

    total_busy = sum(machine_busy.values())
    machine_count = max(1, len(machine_busy))
    machine_utilization = total_busy / (machine_count * cmax) if cmax > 0 else 0.0
    loads = list(machine_busy.values())
    load_balance_std = pstdev(loads) if len(loads) > 1 else 0.0
    split_jobs = [job for job in jobs if len(job.subtasks) > 1]
    split_task_ratio = len(split_jobs) / len(jobs) if jobs else 0.0
    total_split_count = sum(max(0, len(job.subtasks) - 1) for job in jobs)

    return {
        "Cmax_roll": cmax,
        "average_completion_time": avg_completion,
        "average_waiting_time": avg_waiting,
        "machine_utilization": machine_utilization,
        "load_balance_std": load_balance_std,
        "split_task_ratio": split_task_ratio,
        "total_split_count": float(total_split_count),
    }

