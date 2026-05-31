"""Stage A diagnostic for the value of task splitting."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

from instance_manager import SIZES, SPLITS, ensure_fixed_dataset, load_fixed_instances
from src.envs.rolling_scheduling_env import RollingSchedulingEnv
from src.evaluation.metrics import compute_metrics


RESULT_DIR = Path("data/results/stage_A")
OUTPUT_CSV = RESULT_DIR / "split_effect_summary.csv"
ORDERINGS = ["FIFO", "GreedyECT"]
SPLIT_STRATEGIES = [
    "NoSplit",
    "MaxSplit",
    "EqualSplit",
    "SpeedRatioSplit",
    "GreedyECTSplit",
    "RandomSplit",
    "OracleSplitDebug",
]
METRICS = [
    "Cmax_roll",
    "average_completion_time",
    "average_waiting_time",
    "machine_utilization",
    "split_task_ratio",
    "total_split_count",
    "inference_time",
]


class EqualSplitEnv(RollingSchedulingEnv):
    def _compute_split_ratios(self, job_id: str, selected_machines: List[str]) -> Dict[str, float]:
        ratio = 1.0 / len(selected_machines)
        return {machine_id: ratio for machine_id in selected_machines}


def _max_feasible(env: RollingSchedulingEnv, job_id: str) -> int:
    job = env.job_by_id[job_id]
    return max(1, min(job.max_split_num, len(job.candidate_machines)))


def _select_job(env: RollingSchedulingEnv, ordering: str) -> str:
    jobs = env.get_schedulable_jobs()
    if ordering == "FIFO":
        return min(jobs, key=lambda j: (env.job_by_id[j].release_time, j))
    return min(jobs, key=lambda j: (_completion_if_scheduled(env, j, _max_feasible(env, j)), j))


def _completion_if_scheduled(env: RollingSchedulingEnv, job_id: str, split_num: int) -> float:
    trial = env.clone()
    _, _, _, info = trial.step((job_id, split_num))
    return float(info["job_completion_time"])


def _choose_split(env: RollingSchedulingEnv, job_id: str, strategy: str, rng: random.Random) -> int:
    max_split = _max_feasible(env, job_id)
    if strategy == "NoSplit":
        return 1
    if strategy in {"MaxSplit", "EqualSplit", "SpeedRatioSplit", "GreedyECTSplit"}:
        return max_split
    if strategy == "RandomSplit":
        return rng.randint(1, max_split)
    if strategy == "OracleSplitDebug":
        return min(range(1, max_split + 1), key=lambda s: _cmax_if_scheduled(env, job_id, s))
    return min(range(1, max_split + 1), key=lambda s: _completion_if_scheduled(env, job_id, s))


def _cmax_if_scheduled(env: RollingSchedulingEnv, job_id: str, split_num: int) -> float:
    trial = env.clone()
    trial.step((job_id, split_num))
    return float(trial.current_cmax)


def run_split_strategy(instance, ordering: str, strategy: str, seed: int = 42) -> Dict:
    rng = random.Random(seed)
    env_cls = EqualSplitEnv if strategy == "EqualSplit" else RollingSchedulingEnv
    env = env_cls(instance)
    env.reset(instance)
    start = time.perf_counter()
    while not env.is_done():
        job_id = _select_job(env, ordering)
        split_num = _choose_split(env, job_id, strategy, rng)
        env.step((job_id, split_num))
    metrics = compute_metrics(env)
    metrics["inference_time"] = time.perf_counter() - start
    return metrics


def evaluate_split_effect(size: str, split: str, seed: int = 42) -> List[Dict]:
    ensure_fixed_dataset([size], [split])
    rows = []
    for instance in load_fixed_instances(size, split):
        for ordering in ORDERINGS:
            for strategy in SPLIT_STRATEGIES:
                metrics = run_split_strategy(instance, ordering, strategy, seed=seed)
                rows.append(
                    {
                        "size": size,
                        "split": split,
                        "ordering": ordering,
                        "split_strategy": strategy,
                        "seed": getattr(instance, "seed", ""),
                        "instance_id": getattr(instance, "instance_id", instance.name),
                        **metrics,
                    }
                )
    return rows


def _write_summary(rows: Iterable[Dict]) -> None:
    rows = list(rows)
    grouped: Dict[tuple[str, str, str, str], List[Dict]] = {}
    for row in rows:
        grouped.setdefault((row["size"], row["split"], row["ordering"], row["split_strategy"]), []).append(row)

    output_rows = []
    for (size, split, ordering, strategy), group in sorted(grouped.items()):
        out = {
            "size": size,
            "split": split,
            "ordering": ordering,
            "split_strategy": strategy,
            "num_instances": len(group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        output_rows.append(out)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["size", "split", "ordering", "split_strategy", "num_instances"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = evaluate_split_effect(args.size, args.split, seed=args.seed)
    _write_summary(rows)
    print(f"Saved split-effect summary to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
