"""Core data structures for rolling scheduling with task splitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


ProcessType = str
JobId = str
MachineId = str


@dataclass(frozen=True)
class Job:
    job_id: JobId
    process_type: ProcessType
    release_time: float
    candidate_machines: List[MachineId]
    max_split_num: int


@dataclass(frozen=True)
class Machine:
    machine_id: MachineId
    process_type: ProcessType
    speed_factor: float


@dataclass
class SchedulingInstance:
    jobs: List[Job]
    machines: List[Machine]
    processing_time: Dict[JobId, Dict[MachineId, float]]
    rolling_period_length: float
    num_periods: int
    process_types: List[ProcessType] = field(default_factory=lambda: ["sl", "cu", "co"])
    name: str = ""

    @property
    def job_by_id(self) -> Dict[JobId, Job]:
        return {job.job_id: job for job in self.jobs}

    @property
    def machine_by_id(self) -> Dict[MachineId, Machine]:
        return {machine.machine_id: machine for machine in self.machines}


@dataclass
class SubTask:
    job_id: JobId
    machine_id: MachineId
    process_type: ProcessType
    ratio: float
    start_time: float
    duration: float
    completion_time: float


@dataclass
class JobSchedule:
    job_id: JobId
    process_type: ProcessType
    release_time: float
    start_time: float
    completion_time: float
    subtasks: List[SubTask]

