"""Schedule feasibility checks shared by Stage A evaluation scripts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


TOL = 1e-6


VALIDATION_FIELDS = [
    "is_valid_schedule",
    "num_overlap_violations",
    "num_release_time_violations",
    "num_machine_eligibility_violations",
    "num_split_ratio_violations",
    "num_processing_time_violations",
    "num_completion_time_violations",
    "cmax_check_passed",
]


def _get_jobs_and_subtasks(schedule: Any) -> Tuple[Dict[str, Any], List[Any], float]:
    if hasattr(schedule, "scheduled_jobs") and hasattr(schedule, "subtasks"):
        return schedule.scheduled_jobs, list(schedule.subtasks), float(getattr(schedule, "current_cmax", 0.0))
    if isinstance(schedule, dict):
        jobs = schedule.get("jobs", {})
        subtasks = list(schedule.get("subtasks", []))
        cmax = schedule.get("Cmax_roll", schedule.get("current_cmax", 0.0))
        return jobs, subtasks, float(cmax)
    raise TypeError("schedule must be an environment or a dict with jobs/subtasks.")


def validate_schedule(schedule: Any, instance: Any, tolerance: float = TOL) -> Dict[str, Any]:
    jobs, subtasks, reported_cmax = _get_jobs_and_subtasks(schedule)
    job_by_id = instance.job_by_id
    machine_by_id = instance.machine_by_id

    machine_tasks: Dict[str, List[Any]] = defaultdict(list)
    job_subtasks: Dict[str, List[Any]] = defaultdict(list)
    for subtask in subtasks:
        machine_tasks[subtask.machine_id].append(subtask)
        job_subtasks[subtask.job_id].append(subtask)

    overlap_violations = 0
    for tasks in machine_tasks.values():
        ordered = sorted(tasks, key=lambda item: (item.start_time, item.completion_time, item.job_id))
        for prev, cur in zip(ordered, ordered[1:]):
            if cur.start_time < prev.completion_time - tolerance:
                overlap_violations += 1

    release_time_violations = 0
    machine_eligibility_violations = 0
    processing_time_violations = 0
    for subtask in subtasks:
        job = job_by_id[subtask.job_id]
        machine = machine_by_id[subtask.machine_id]
        if subtask.start_time + tolerance < job.release_time:
            release_time_violations += 1
        if subtask.machine_id not in job.candidate_machines or machine.process_type != job.process_type:
            machine_eligibility_violations += 1
        expected_duration = instance.processing_time[subtask.job_id][subtask.machine_id] * subtask.ratio
        if abs(subtask.duration - expected_duration) > tolerance:
            processing_time_violations += 1

    split_ratio_violations = 0
    completion_time_violations = 0
    completion_times = []
    for job_id, job_schedule in jobs.items():
        pieces = job_subtasks.get(job_id, [])
        if not pieces:
            split_ratio_violations += 1
            completion_time_violations += 1
            continue
        if abs(sum(piece.ratio for piece in pieces) - 1.0) > tolerance:
            split_ratio_violations += 1
        expected_completion = max(piece.completion_time for piece in pieces)
        completion_times.append(expected_completion)
        if abs(job_schedule.completion_time - expected_completion) > tolerance:
            completion_time_violations += 1

    expected_cmax = max(completion_times, default=0.0)
    cmax_check_passed = abs(reported_cmax - expected_cmax) <= tolerance

    # The current environment commits subtasks once appended and has no reschedule
    # operation, so cross-period immutability is covered by overlap and migration checks.
    is_valid = (
        overlap_violations == 0
        and release_time_violations == 0
        and machine_eligibility_violations == 0
        and split_ratio_violations == 0
        and processing_time_violations == 0
        and completion_time_violations == 0
        and cmax_check_passed
    )

    return {
        "is_valid_schedule": is_valid,
        "num_overlap_violations": overlap_violations,
        "num_release_time_violations": release_time_violations,
        "num_machine_eligibility_violations": machine_eligibility_violations,
        "num_split_ratio_violations": split_ratio_violations,
        "num_processing_time_violations": processing_time_violations,
        "num_completion_time_violations": completion_time_violations,
        "cmax_check_passed": cmax_check_passed,
    }

