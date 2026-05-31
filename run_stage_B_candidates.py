"""Stage B candidate-set ablation runner.

This script is intended for later full experiments. During implementation, use
``--max_instances 1`` for a dry-run only.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

from candidate_generator import CANDIDATE_MODES, generate_candidates
from instance_manager import SIZES, SPLITS, ensure_fixed_dataset, load_fixed_instances
from schedule_validator import VALIDATION_FIELDS, validate_schedule
from src.baselines.heuristics import choose_split_num, estimated_completion_time, lookahead_score
from src.envs.rolling_scheduling_env import RollingSchedulingEnv
from src.evaluation.metrics import compute_metrics


RESULT_DIR = Path("data/results/stage_B")
DETAIL_CSV = RESULT_DIR / "candidate_ablation.csv"
SUMMARY_CSV = RESULT_DIR / "candidate_ablation_summary.csv"

METHODS = [
    "all_legal_random",
    "hybrid_topk_random",
    "fifo_topk_greedy",
    "greedy_ect_topk_greedy",
    "lookahead_topk_greedy",
    "hybrid_topk_greedy",
    "hybrid_topk_oracle_debug",
]
METRICS = [
    "Cmax_roll",
    "average_completion_time",
    "average_waiting_time",
    "machine_utilization",
    "load_balance_std",
    "split_task_ratio",
    "total_split_count",
    "inference_time",
]
ROW_FIELDS = [
    "method",
    "size",
    "split",
    "top_k",
    "instance_id",
    *METRICS,
    *VALIDATION_FIELDS,
]


def _mode_for_method(method: str) -> str:
    if method == "all_legal_random":
        return "all"
    if method.startswith("fifo_topk"):
        return "fifo_topk"
    if method.startswith("greedy_ect_topk"):
        return "greedy_ect_topk"
    if method.startswith("lookahead_topk"):
        return "lookahead_topk"
    return "hybrid_topk"


def _local_cmax_if_scheduled(env: RollingSchedulingEnv, job_id: str) -> float:
    return max(float(env.current_cmax), estimated_completion_time(env, job_id))


def _select_from_candidates(env: RollingSchedulingEnv, method: str, candidates: List[str], rng: random.Random) -> str:
    if not candidates:
        raise RuntimeError("Candidate set is empty and fallback failed.")
    if method.endswith("_random"):
        return rng.choice(candidates)
    if method == "lookahead_topk_greedy":
        return min(candidates, key=lambda j: (lookahead_score(env, j), j))
    if method == "hybrid_topk_oracle_debug":
        return min(candidates, key=lambda j: (_local_cmax_if_scheduled(env, j), j))
    return min(candidates, key=lambda j: (estimated_completion_time(env, j), j))


def run_candidate_method(instance, method: str, top_k: int = 5, seed: int = 42) -> Dict:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Expected one of {METHODS}.")

    rng = random.Random(seed)
    env = RollingSchedulingEnv(instance)
    env.reset(instance)
    candidate_mode = _mode_for_method(method)
    start = time.perf_counter()
    while not env.is_done():
        candidates = generate_candidates(
            env,
            candidate_mode=candidate_mode,
            top_k=top_k,
            allow_duplicate=False,
            fallback_to_all=True,
        )
        job_id = _select_from_candidates(env, method, candidates, rng)
        env.step((job_id, choose_split_num(env, job_id)))
    metrics = compute_metrics(env)
    metrics["inference_time"] = time.perf_counter() - start
    metrics.update(validate_schedule(env, instance))
    return metrics


def evaluate_candidate_ablation(
    size: str,
    split: str,
    top_k: int,
    seed: int = 42,
    max_instances: int | None = None,
) -> List[Dict]:
    ensure_fixed_dataset([size], [split])
    instances = load_fixed_instances(size, split)
    if max_instances is not None:
        instances = instances[: max(0, max_instances)]

    rows: List[Dict] = []
    for instance in instances:
        for method in METHODS:
            metrics = run_candidate_method(instance, method, top_k=top_k, seed=seed)
            rows.append(
                {
                    "method": method,
                    "size": size,
                    "split": split,
                    "top_k": top_k,
                    "instance_id": getattr(instance, "instance_id", instance.name),
                    **metrics,
                }
            )
    return rows


def write_details(rows: Iterable[Dict], output_path: Path = DETAIL_CSV) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: Iterable[Dict], output_path: Path = SUMMARY_CSV) -> List[Dict]:
    rows = list(rows)
    grouped: Dict[tuple[str, str, str, int], List[Dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["size"], row["split"], int(row["top_k"])), []).append(row)

    summary_rows = []
    for (method, size, split, top_k), group in sorted(grouped.items()):
        out = {"method": method, "size": size, "split": split, "top_k": top_k, "num_instances": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out["valid_ratio"] = mean(1.0 if row["is_valid_schedule"] else 0.0 for row in group)
        out["cmax_check_pass_ratio"] = mean(1.0 if row["cmax_check_passed"] else 0.0 for row in group)
        summary_rows.append(out)

    fieldnames = ["method", "size", "split", "top_k", "num_instances"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    fieldnames.extend(["valid_ratio", "cmax_check_pass_ratio"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_instances", type=int, default=None)
    args = parser.parse_args()

    rows = evaluate_candidate_ablation(
        args.size,
        args.split,
        top_k=args.top_k,
        seed=args.seed,
        max_instances=args.max_instances,
    )
    write_details(rows)
    summary_rows = write_summary(rows)
    print(f"Saved {len(rows)} rows to {DETAIL_CSV}")
    print(f"Saved {len(summary_rows)} summary rows to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()

