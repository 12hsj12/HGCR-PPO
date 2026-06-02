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

from candidate_generator import CANDIDATE_MODES, fifo_ranked, first_rule_jobs, generate_candidates
from instance_manager import SIZES, SPLITS, ensure_fixed_dataset, load_fixed_instances
from schedule_validator import VALIDATION_FIELDS, validate_schedule
from src.baselines.heuristics import choose_split_num, estimated_completion_time, lookahead_score
from src.envs.rolling_scheduling_env import RollingSchedulingEnv
from src.evaluation.metrics import compute_metrics
from utils.experiment_io import make_result_path, make_run_id, rebuild_all_summary, update_latest_file


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
    "hybrid_topk_fifo_select",
    "hybrid_topk_lookahead_select",
    "hybrid_topk_min_cmax_select",
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
DIAGNOSTIC_FIELDS = [
    "candidate_size_mean",
    "candidate_size_std",
    "fifo_first_in_candidate_ratio",
    "greedy_first_in_candidate_ratio",
    "lookahead_first_in_candidate_ratio",
    "minload_first_in_candidate_ratio",
    "selected_from_fifo_ratio",
    "selected_from_greedy_ratio",
    "selected_from_lookahead_ratio",
    "selected_from_minload_ratio",
]
ROW_FIELDS = [
    "size",
    "split",
    "top_k",
    "method",
    "instance_id",
    *METRICS,
    *DIAGNOSTIC_FIELDS,
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


def _rollout_cmax(env: RollingSchedulingEnv, first_job_id: str, rollout_policy: str) -> float:
    trial = env.clone()
    trial.step((first_job_id, choose_split_num(trial, first_job_id)))
    while not trial.is_done():
        jobs = trial.get_schedulable_jobs()
        if rollout_policy == "lookahead":
            job_id = min(jobs, key=lambda j: (lookahead_score(trial, j), j))
        else:
            job_id = min(jobs, key=lambda j: (trial.job_by_id[j].release_time, j))
        trial.step((job_id, choose_split_num(trial, job_id)))
    return float(trial.current_cmax)


def _select_from_candidates(
    env: RollingSchedulingEnv,
    method: str,
    candidates: List[str],
    rng: random.Random,
    oracle_rollout_policy: str = "fifo",
) -> str:
    if not candidates:
        raise RuntimeError("Candidate set is empty and fallback failed.")
    if method.endswith("_random"):
        return rng.choice(candidates)
    if method == "hybrid_topk_fifo_select":
        fifo_order = {job_id: idx for idx, job_id in enumerate(fifo_ranked(env))}
        return min(candidates, key=lambda j: (fifo_order.get(j, 10**9), j))
    if method in {"lookahead_topk_greedy", "hybrid_topk_lookahead_select"}:
        return min(candidates, key=lambda j: (lookahead_score(env, j), j))
    if method == "hybrid_topk_min_cmax_select":
        return min(candidates, key=lambda j: (_local_cmax_if_scheduled(env, j), j))
    if method == "hybrid_topk_oracle_debug":
        return min(candidates, key=lambda j: (_rollout_cmax(env, j, oracle_rollout_policy), j))
    return min(candidates, key=lambda j: (estimated_completion_time(env, j), j))


def _empty_diagnostics() -> Dict[str, List[float]]:
    return {field: [] for field in DIAGNOSTIC_FIELDS}


def _record_diagnostics(diagnostics: Dict[str, List[float]], env, candidates: List[str], selected_job: str) -> None:
    firsts = first_rule_jobs(env)
    candidate_set = set(candidates)
    diagnostics["candidate_size_mean"].append(float(len(candidates)))
    diagnostics["candidate_size_std"].append(float(len(candidates)))
    for field, key in [
        ("fifo_first_in_candidate_ratio", "fifo"),
        ("greedy_first_in_candidate_ratio", "greedy"),
        ("lookahead_first_in_candidate_ratio", "lookahead"),
        ("minload_first_in_candidate_ratio", "minload"),
        ("selected_from_fifo_ratio", "fifo"),
        ("selected_from_greedy_ratio", "greedy"),
        ("selected_from_lookahead_ratio", "lookahead"),
        ("selected_from_minload_ratio", "minload"),
    ]:
        first = firsts[key]
        if field.endswith("_in_candidate_ratio"):
            diagnostics[field].append(1.0 if first is not None and first in candidate_set else 0.0)
        else:
            diagnostics[field].append(1.0 if first is not None and selected_job == first else 0.0)


def _finalize_diagnostics(diagnostics: Dict[str, List[float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    sizes = diagnostics["candidate_size_mean"]
    out["candidate_size_mean"] = mean(sizes) if sizes else 0.0
    out["candidate_size_std"] = pstdev(sizes) if len(sizes) > 1 else 0.0
    for field in DIAGNOSTIC_FIELDS:
        if field in {"candidate_size_mean", "candidate_size_std"}:
            continue
        values = diagnostics[field]
        out[field] = mean(values) if values else 0.0
    return out


def run_candidate_method(
    instance,
    method: str,
    top_k: int = 5,
    seed: int = 42,
    oracle_rollout_policy: str = "fifo",
) -> Dict:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Expected one of {METHODS}.")

    rng = random.Random(seed)
    env = RollingSchedulingEnv(instance)
    env.reset(instance)
    candidate_mode = _mode_for_method(method)
    diagnostics = _empty_diagnostics()
    start = time.perf_counter()
    while not env.is_done():
        candidates = generate_candidates(
            env,
            candidate_mode=candidate_mode,
            top_k=top_k,
            allow_duplicate=False,
            fallback_to_all=True,
        )
        job_id = _select_from_candidates(env, method, candidates, rng, oracle_rollout_policy=oracle_rollout_policy)
        _record_diagnostics(diagnostics, env, candidates, job_id)
        env.step((job_id, choose_split_num(env, job_id)))
    metrics = compute_metrics(env)
    metrics["inference_time"] = time.perf_counter() - start
    metrics.update(_finalize_diagnostics(diagnostics))
    metrics.update(validate_schedule(env, instance))
    return metrics


def evaluate_candidate_ablation(
    size: str,
    split: str,
    top_k: int,
    seed: int = 42,
    max_instances: int | None = None,
    oracle_rollout_policy: str = "fifo",
) -> List[Dict]:
    ensure_fixed_dataset([size], [split])
    instances = load_fixed_instances(size, split)
    if max_instances is not None:
        instances = instances[: max(0, max_instances)]

    rows: List[Dict] = []
    for instance in instances:
        for method in METHODS:
            metrics = run_candidate_method(
                instance,
                method,
                top_k=top_k,
                seed=seed,
                oracle_rollout_policy=oracle_rollout_policy,
            )
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


def result_paths(size: str, split: str, top_k: int, run_id: str, overwrite: bool) -> tuple[Path, Path]:
    tokens = [size, split, f"topk{int(top_k)}", f"runid{run_id}"]
    return (
        make_result_path(RESULT_DIR, "candidate_ablation", tokens, run_id=None, overwrite=overwrite),
        make_result_path(RESULT_DIR, "candidate_ablation_summary", tokens, run_id=None, overwrite=overwrite),
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
        for field in DIAGNOSTIC_FIELDS:
            values = [float(row.get(field, 0.0) or 0.0) for row in group]
            out[field] = mean(values) if values else 0.0
        out["valid_ratio"] = mean(1.0 if _truthy(row["is_valid_schedule"]) else 0.0 for row in group)
        out["cmax_check_pass_ratio"] = mean(1.0 if _truthy(row["cmax_check_passed"]) else 0.0 for row in group)
        summary_rows.append(out)

    fieldnames = summary_fields()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def summary_fields() -> List[str]:
    fieldnames = ["method", "size", "split", "top_k", "num_instances"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    fieldnames.extend(DIAGNOSTIC_FIELDS)
    fieldnames.extend(["valid_ratio", "cmax_check_pass_ratio"])
    return fieldnames


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def write_all_outputs(result_dir: Path = RESULT_DIR) -> tuple[Path, Path]:
    rebuild_all_summary(
        result_dir=result_dir,
        pattern="candidate_ablation_*_topk*.csv",
        output_path=ALL_DETAIL_CSV,
        fieldnames=ROW_FIELDS,
        exclude_names={
            "candidate_ablation_latest.csv",
            "candidate_ablation_all.csv",
            "candidate_ablation_summary_latest.csv",
            "candidate_ablation_summary_all.csv",
        },
        exclude_prefixes=("candidate_ablation_summary_",),
        required_substrings=("_runid",),
    )
    rebuild_all_summary(
        result_dir=result_dir,
        pattern="candidate_ablation_summary_*_topk*.csv",
        output_path=ALL_SUMMARY_CSV,
        fieldnames=summary_fields(),
        exclude_names={
            "candidate_ablation_summary_latest.csv",
            "candidate_ablation_summary_all.csv",
        },
        required_substrings=("_runid",),
    )
    return ALL_DETAIL_CSV, ALL_SUMMARY_CSV


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--oracle_rollout_policy", choices=["fifo", "lookahead"], default="fifo")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_id = make_run_id(args.run_id)

    rows = evaluate_candidate_ablation(
        args.size,
        args.split,
        top_k=args.top_k,
        seed=args.seed,
        max_instances=args.max_instances,
        oracle_rollout_policy=args.oracle_rollout_policy,
    )
    detail_path, summary_path = result_paths(args.size, args.split, args.top_k, run_id, args.overwrite)
    write_details(rows, detail_path)
    update_latest_file(rows, LATEST_DETAIL_CSV, ROW_FIELDS)
    summary_rows = write_summary(rows, summary_path)
    update_latest_file(summary_rows, LATEST_SUMMARY_CSV, summary_fields())
    all_detail_path, all_summary_path = write_all_outputs()

    print(f"Saved {len(rows)} rows to {detail_path}")
    print(f"Saved {len(summary_rows)} summary rows to {summary_path}")
    print(f"Updated latest files: {LATEST_DETAIL_CSV}, {LATEST_SUMMARY_CSV}")
    print(f"Updated all files: {all_detail_path}, {all_summary_path}")


if __name__ == "__main__":
    main()
