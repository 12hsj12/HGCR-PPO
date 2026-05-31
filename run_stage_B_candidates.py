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
LATEST_DETAIL_CSV = RESULT_DIR / "candidate_ablation_latest.csv"
LATEST_SUMMARY_CSV = RESULT_DIR / "candidate_ablation_summary_latest.csv"
ALL_DETAIL_CSV = RESULT_DIR / "candidate_ablation_all.csv"
ALL_SUMMARY_CSV = RESULT_DIR / "candidate_ablation_summary_all.csv"

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
    "size",
    "split",
    "top_k",
    "method",
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


def result_paths(size: str, split: str, top_k: int) -> tuple[Path, Path]:
    suffix = f"{size}_{split}_topk{int(top_k)}"
    return (
        RESULT_DIR / f"candidate_ablation_{suffix}.csv",
        RESULT_DIR / f"candidate_ablation_summary_{suffix}.csv",
    )


def write_details(rows: Iterable[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: Iterable[Dict], output_path: Path) -> List[Dict]:
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


def _is_detail_topk_file(path: Path) -> bool:
    name = path.name
    return (
        name.startswith("candidate_ablation_")
        and "_topk" in name
        and not name.startswith("candidate_ablation_summary_")
        and name not in {LATEST_DETAIL_CSV.name, ALL_DETAIL_CSV.name}
    )


def collect_topk_detail_rows(result_dir: Path = RESULT_DIR) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(result_dir.glob("candidate_ablation_*_topk*.csv")):
        if not _is_detail_topk_file(path):
            continue
        with path.open("r", newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def write_all_outputs(result_dir: Path = RESULT_DIR) -> tuple[Path, Path]:
    rows = collect_topk_detail_rows(result_dir)
    write_details(rows, ALL_DETAIL_CSV)
    write_summary(rows, ALL_SUMMARY_CSV)
    return ALL_DETAIL_CSV, ALL_SUMMARY_CSV


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
    detail_path, summary_path = result_paths(args.size, args.split, args.top_k)
    write_details(rows, detail_path)
    write_details(rows, LATEST_DETAIL_CSV)
    summary_rows = write_summary(rows, summary_path)
    write_summary(rows, LATEST_SUMMARY_CSV)
    all_detail_path, all_summary_path = write_all_outputs()

    print(f"Saved {len(rows)} rows to {detail_path}")
    print(f"Saved {len(summary_rows)} summary rows to {summary_path}")
    print(f"Updated latest files: {LATEST_DETAIL_CSV}, {LATEST_SUMMARY_CSV}")
    print(f"Updated all files: {all_detail_path}, {all_summary_path}")


if __name__ == "__main__":
    main()
