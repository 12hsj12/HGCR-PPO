"""Dynamic rolling scenario generator for HGCR-PPO.

The generator derives dynamic training/evaluation scenarios from the fixed
Stage A datasets without mutating or overwriting those datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import uuid
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

import torch

from instance_manager import SIZES, SPLITS, ensure_fixed_dataset, load_fixed_instances


LEVELS = ["low", "medium", "high"]
ARRIVAL_FRACTION = {"low": 0.35, "medium": 0.60, "high": 0.85}
CARRYOVER_FRACTION = {"low": 0.10, "medium": 0.25, "high": 0.40}
INITIAL_LOAD_LEVEL = {"low": 0.05, "medium": 0.15, "high": 0.30}
ROOT = Path("data/generated/dynamic_scenarios/runs")


def make_run_id(args) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"dyn_{args.size}_{args.split}_arr-{args.arrival_intensity}_car-{args.carryover_ratio}"
        f"_n{args.num_scenarios}_s{args.seed}_{stamp}_{uuid.uuid4().hex[:8]}"
    )


def scenario_paths(run_id: str) -> Dict[str, Path]:
    run_dir = ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return {
        "run_dir": run_dir,
        "scenarios": run_dir / f"scenarios__{run_id}.pt",
        "summary": run_dir / f"scenario_summary__{run_id}.csv",
        "manifest": run_dir / f"manifest__{run_id}.json",
    }


def _scheduled_job_counts(total_jobs: int, arrival_intensity: str, carryover_ratio: str) -> tuple[int, int]:
    carryover = max(1, int(round(total_jobs * CARRYOVER_FRACTION[carryover_ratio])))
    arriving = max(1, int(round(total_jobs * ARRIVAL_FRACTION[arrival_intensity])))
    if carryover + arriving > total_jobs:
        arriving = max(1, total_jobs - carryover)
    return carryover, arriving


def _machine_initial_available(instance, rng: random.Random, machine_initial_load: str) -> Dict[str, float]:
    period = float(getattr(instance, "rolling_period_length", 1.0) or 1.0)
    scale = INITIAL_LOAD_LEVEL[machine_initial_load]
    return {machine.machine_id: rng.uniform(0.0, period * scale) for machine in instance.machines}


def build_scenario(
    base_instance,
    scenario_index: int,
    seed: int,
    arrival_intensity: str,
    carryover_ratio: str,
    processing_time_noise: float = 0.0,
    machine_initial_load: str = "low",
):
    rng = random.Random(seed)
    instance = deepcopy(base_instance)
    jobs = sorted(instance.jobs, key=lambda job: (job.release_time, job.job_id))
    total_jobs = len(jobs)
    carryover_count, arriving_count = _scheduled_job_counts(total_jobs, arrival_intensity, carryover_ratio)
    carryover_jobs = [job.job_id for job in jobs[:carryover_count]]
    remaining = jobs[carryover_count:]
    rng.shuffle(remaining)
    arrival_jobs = sorted(remaining[:arriving_count], key=lambda job: job.job_id)
    arrival_job_ids = [job.job_id for job in arrival_jobs]

    period = float(getattr(instance, "rolling_period_length", 1.0) or 1.0)
    release_updates: Dict[str, float] = {}
    for job_id in carryover_jobs:
        release_updates[job_id] = 0.0
    for idx, job in enumerate(arrival_jobs):
        base = period * (idx + 1) / max(1, arriving_count + 1)
        jitter = rng.uniform(-0.08 * period, 0.08 * period)
        release_updates[job.job_id] = max(0.0, min(period, base + jitter))
    for job in jobs:
        release_updates.setdefault(job.job_id, period + float(job.release_time))

    instance.jobs = [replace(job, release_time=float(release_updates[job.job_id])) for job in instance.jobs]
    instance.name = f"{getattr(base_instance, 'instance_id', getattr(base_instance, 'name', 'fixed'))}__dyn_{scenario_index:04d}"
    instance.instance_id = instance.name
    instance.scenario_type = "dynamic"
    instance.arrival_intensity = arrival_intensity
    instance.carryover_ratio = carryover_ratio
    instance.processing_time_noise = processing_time_noise
    instance.machine_initial_load = machine_initial_load
    instance.release_time = {job.job_id: job.release_time for job in instance.jobs}

    machine_available = _machine_initial_available(instance, rng, machine_initial_load)
    return {
        "scenario_id": f"scenario_{scenario_index:04d}_seed_{seed}",
        "scenario_type": "dynamic",
        "seed": seed,
        "base_instance_id": getattr(base_instance, "instance_id", getattr(base_instance, "name", "")),
        "size": getattr(base_instance, "size", ""),
        "split": getattr(base_instance, "split", ""),
        "arrival_intensity": arrival_intensity,
        "carryover_ratio": carryover_ratio,
        "processing_time_noise": processing_time_noise,
        "machine_initial_load": machine_initial_load,
        "new_arrival_batch": arrival_job_ids,
        "carryover_batch": carryover_jobs,
        "machine_initial_available_time": machine_available,
        "instance": instance,
    }


def generate_dynamic_scenarios(
    size: str,
    split: str,
    num_scenarios: int,
    seed: int,
    arrival_intensity: str,
    carryover_ratio: str,
    processing_time_noise: float = 0.0,
    machine_initial_load: str = "low",
) -> List[dict]:
    ensure_fixed_dataset([size], [split])
    fixed_instances = load_fixed_instances(size, split)
    rng = random.Random(seed)
    scenarios = []
    for idx in range(num_scenarios):
        base = fixed_instances[idx % len(fixed_instances)]
        scenario_seed = seed * 100000 + idx
        if idx >= len(fixed_instances):
            base = rng.choice(fixed_instances)
        scenarios.append(
            build_scenario(
                base,
                idx,
                scenario_seed,
                arrival_intensity,
                carryover_ratio,
                processing_time_noise=processing_time_noise,
                machine_initial_load=machine_initial_load,
            )
        )
    return scenarios


def write_summary(scenarios: Iterable[dict], path: Path) -> None:
    rows = []
    for scenario in scenarios:
        machine_loads = list(scenario["machine_initial_available_time"].values())
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "base_instance_id": scenario["base_instance_id"],
                "size": scenario["size"],
                "split": scenario["split"],
                "arrival_intensity": scenario["arrival_intensity"],
                "carryover_ratio": scenario["carryover_ratio"],
                "processing_time_noise": scenario["processing_time_noise"],
                "machine_initial_load": scenario["machine_initial_load"],
                "num_new_arrivals": len(scenario["new_arrival_batch"]),
                "num_carryover_jobs": len(scenario["carryover_batch"]),
                "machine_initial_available_mean": mean(machine_loads) if machine_loads else 0.0,
            }
        )
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_scenario_run(args, run_id: str | None = None) -> Dict[str, Path]:
    run_id = run_id or make_run_id(args)
    paths = scenario_paths(run_id)
    scenarios = generate_dynamic_scenarios(
        args.size,
        args.split,
        args.num_scenarios,
        args.seed,
        args.arrival_intensity,
        args.carryover_ratio,
        processing_time_noise=args.processing_time_noise,
        machine_initial_load=args.machine_initial_load,
    )
    torch.save(scenarios, paths["scenarios"])
    write_summary(scenarios, paths["summary"])
    paths["manifest"].write_text(
        json.dumps(
            {
                "run_id": run_id,
                "args": vars(args),
                "num_scenarios": len(scenarios),
                "output_files": {key: str(value) for key, value in paths.items() if key != "run_dir"},
                "python_version": sys.version,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, default="small")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--arrival_intensity", choices=LEVELS, default="medium")
    parser.add_argument("--carryover_ratio", choices=LEVELS, default="medium")
    parser.add_argument("--processing_time_noise", type=float, choices=[0.0, 0.1, 0.2], default=0.0)
    parser.add_argument("--machine_initial_load", choices=LEVELS, default="low")
    parser.add_argument("--num_scenarios", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    paths = save_scenario_run(args)
    print(f"Saved dynamic scenarios to {paths['run_dir']}")


if __name__ == "__main__":
    main()
