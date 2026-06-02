"""Evaluate Stage C MLP-BC and MLP-Ranker policies on HybridTopK candidates."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

import torch

from candidate_generator import fifo_ranked
from instance_manager import SIZES, SPLITS, ensure_fixed_dataset, load_fixed_instances
from mlp_models import load_checkpoint
from schedule_validator import VALIDATION_FIELDS, validate_schedule
from src.baselines.heuristics import choose_split_num
from src.envs.rolling_scheduling_env import RollingSchedulingEnv
from src.evaluation.metrics import compute_metrics
from stage_c_utils import (
    best_candidate_index,
    extract_candidate_features,
    fifo_first,
    hybrid_candidates,
    oracle_cmax_per_candidate,
)
from utils.experiment_io import (
    make_result_path,
    make_run_id,
    rebuild_all_summary,
    sanitize_token,
    save_csv_no_overwrite,
    update_latest_file,
)


RESULT_DIR = Path("data/results/stage_C")
METHODS = [
    "hybrid_topk_random",
    "hybrid_topk_fifo_select",
    "mlp_bc",
    "mlp_ranker",
    "oracle_debug",
]
DEFAULT_RULE_METHODS = [
    "hybrid_topk_random",
    "hybrid_topk_fifo_select",
    "oracle_debug",
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
    "action_match_oracle_ratio",
    "action_match_fifo_ratio",
]
ROW_FIELDS = [
    "method",
    "size",
    "split",
    "top_k",
    "model_tag",
    "instance_id",
    *METRICS,
    *VALIDATION_FIELDS,
]


def _model_select(model, env, candidates: List[str]) -> str:
    features = torch.tensor([extract_candidate_features(env, candidates)], dtype=torch.float32)
    with torch.no_grad():
        scores = model(features)[0]
    return candidates[int(scores.argmax().item())]


def _select_job(
    method: str,
    env,
    candidates: List[str],
    rng: random.Random,
    oracle_rollout_policy: str,
    model=None,
    oracle_cmax: List[float] | None = None,
) -> str:
    if method == "hybrid_topk_random":
        return rng.choice(candidates)
    if method == "hybrid_topk_fifo_select":
        fifo_order = {job_id: idx for idx, job_id in enumerate(fifo_ranked(env))}
        return min(candidates, key=lambda j: (fifo_order.get(j, 10**9), j))
    if method in {"mlp_bc", "mlp_ranker"}:
        if model is None:
            if method == "mlp_bc":
                raise ValueError(
                    "mlp_bc requires --bc_model_path. Please train MLP-BC first or remove mlp_bc from --methods."
                )
            raise ValueError(
                "mlp_ranker requires --ranker_model_path. Please train MLP-Ranker first or remove mlp_ranker from --methods."
            )
        return _model_select(model, env, candidates)
    if method == "oracle_debug":
        values = oracle_cmax or oracle_cmax_per_candidate(env, candidates, rollout_policy=oracle_rollout_policy)
        return candidates[best_candidate_index(values)]
    raise ValueError(f"Unknown method {method!r}.")


def run_method(
    instance,
    method: str,
    top_k: int,
    seed: int,
    oracle_rollout_policy: str,
    model=None,
) -> Dict:
    rng = random.Random(seed)
    env = RollingSchedulingEnv(instance)
    env.reset(instance)
    oracle_matches = []
    fifo_matches = []
    start = time.perf_counter()

    while not env.is_done():
        candidates = hybrid_candidates(env, top_k)
        oracle_cmax = oracle_cmax_per_candidate(env, candidates, rollout_policy=oracle_rollout_policy)
        oracle_job = candidates[best_candidate_index(oracle_cmax)]
        fifo_job = fifo_first(env)
        job_id = _select_job(
            method,
            env,
            candidates,
            rng,
            oracle_rollout_policy=oracle_rollout_policy,
            model=model,
            oracle_cmax=oracle_cmax,
        )
        oracle_matches.append(1.0 if job_id == oracle_job else 0.0)
        fifo_matches.append(1.0 if fifo_job is not None and job_id == fifo_job else 0.0)
        env.step((job_id, choose_split_num(env, job_id)))

    metrics = compute_metrics(env)
    metrics["inference_time"] = time.perf_counter() - start
    metrics["action_match_oracle_ratio"] = mean(oracle_matches) if oracle_matches else 0.0
    metrics["action_match_fifo_ratio"] = mean(fifo_matches) if fifo_matches else 0.0
    metrics.update(validate_schedule(env, instance))
    return metrics


def evaluate_ranker(
    size: str,
    split: str,
    top_k: int,
    model_tag: str,
    bc_model_path: str | None,
    ranker_model_path: str | None,
    methods: List[str],
    max_instances: int | None,
    seed: int,
    oracle_rollout_policy: str,
) -> List[Dict]:
    ensure_fixed_dataset([size], [split])
    instances = load_fixed_instances(size, split)
    if max_instances is not None:
        instances = instances[: max(0, max_instances)]

    models = _load_models_for_methods(methods, bc_model_path, ranker_model_path, size, top_k)
    rows: List[Dict] = []
    for instance in instances:
        for method in methods:
            model = models.get(method)
            metrics = run_method(instance, method, top_k, seed, oracle_rollout_policy, model=model)
            rows.append(
                {
                    "method": method,
                    "size": size,
                    "split": split,
                    "top_k": top_k,
                    "model_tag": model_tag,
                    "instance_id": getattr(instance, "instance_id", instance.name),
                    **metrics,
                }
            )
    return rows


def _load_models_for_methods(
    methods: List[str],
    bc_model_path: str | None,
    ranker_model_path: str | None,
    size: str,
    top_k: int,
) -> Dict[str, object]:
    models: Dict[str, object] = {}
    if "mlp_bc" in methods:
        if not bc_model_path:
            raise ValueError(
                "mlp_bc requires --bc_model_path. Please train MLP-BC first or remove mlp_bc from --methods."
            )
        bc_path = Path(bc_model_path)
        if not bc_path.exists():
            raise FileNotFoundError(_missing_bc_checkpoint_message(bc_path, size, top_k))
        models["mlp_bc"] = load_checkpoint(bc_path)
    if "mlp_ranker" in methods:
        if not ranker_model_path:
            raise ValueError(
                "mlp_ranker requires --ranker_model_path. Please train MLP-Ranker first or remove mlp_ranker from --methods."
            )
        ranker_path = Path(ranker_model_path)
        if not ranker_path.exists():
            raise FileNotFoundError(_missing_ranker_checkpoint_message(ranker_path, size, top_k))
        models["mlp_ranker"] = load_checkpoint(ranker_path)
    return models


def _missing_bc_checkpoint_message(path: Path, size: str, top_k: int) -> str:
    return (
        f"BC checkpoint not found: {path}\n"
        "Please check whether train_mlp_bc.py saved the model under:\n"
        f"checkpoints/stage_C/mlp_bc/{size}_topk{top_k}_runid{{run_id}}/best.pt\n"
        "or use the latest checkpoint:\n"
        f"checkpoints/stage_C/mlp_bc/{size}_topk{top_k}_latest/best.pt"
    )


def _missing_ranker_checkpoint_message(path: Path, size: str, top_k: int) -> str:
    return (
        f"Ranker checkpoint not found: {path}\n"
        "Please check whether train_mlp_ranker.py saved the model under:\n"
        f"checkpoints/stage_C/mlp_ranker/{size}_topk{top_k}_{{loss_type}}_runid{{run_id}}/best.pt\n"
        "or use the latest checkpoint:\n"
        f"checkpoints/stage_C/mlp_ranker/{size}_topk{top_k}_{{loss_type}}_latest/best.pt"
    )


def write_details(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: Iterable[Dict], path: Path) -> None:
    fieldnames = summary_fields()
    summary_rows = summarize_rows(rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def summary_fields() -> List[str]:
    fieldnames = ["method", "size", "split", "top_k", "model_tag", "num_instances"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    fieldnames.extend(["valid_ratio", "cmax_check_pass_ratio"])
    return fieldnames


def result_paths(size: str, split: str, top_k: int, model_tag: str, run_id: str, overwrite: bool) -> tuple[Path, Path]:
    tokens = [size, split, f"topk{top_k}", model_tag, f"runid{run_id}"]
    return (
        make_result_path(RESULT_DIR, "ranker_eval", tokens, run_id="", overwrite=overwrite),
        make_result_path(RESULT_DIR, "ranker_eval_summary", tokens, run_id="", overwrite=overwrite),
    )


def infer_model_tag(methods: List[str], bc_model_path: str | None, ranker_model_path: str | None, model_tag: str | None) -> str:
    if model_tag:
        return sanitize_token(model_tag)
    model_methods = {"mlp_bc", "mlp_ranker"}
    if not any(method in model_methods for method in methods):
        return "rules_only"
    ranker_path = str(ranker_model_path or "").lower()
    if "soft_ce" in ranker_path:
        return "soft_ce"
    if "pairwise" in ranker_path:
        return "pairwise"
    if "mlp_bc" in methods and not ranker_model_path:
        return "bc_only"
    return "manual"


def update_latest_and_all(rows: List[Dict]) -> None:
    latest_detail_path = RESULT_DIR / "ranker_eval_latest.csv"
    latest_summary_path = RESULT_DIR / "ranker_eval_summary_latest.csv"
    update_latest_file(rows, latest_detail_path, ROW_FIELDS)
    summary_rows = summarize_rows(rows)
    update_latest_file(summary_rows, latest_summary_path, summary_fields())
    rebuild_all_summary(
        result_dir=RESULT_DIR,
        pattern="ranker_eval_*_topk*.csv",
        output_path=RESULT_DIR / "ranker_eval_all.csv",
        fieldnames=ROW_FIELDS,
        exclude_names={
            "ranker_eval.csv",
            "ranker_eval_latest.csv",
            "ranker_eval_all.csv",
            "ranker_eval_summary.csv",
            "ranker_eval_summary_latest.csv",
            "ranker_eval_summary_all.csv",
        },
        exclude_prefixes=("ranker_eval_summary_",),
        required_substrings=("_runid",),
    )
    rebuild_all_summary(
        result_dir=RESULT_DIR,
        pattern="ranker_eval_summary_*_topk*.csv",
        output_path=RESULT_DIR / "ranker_eval_summary_all.csv",
        fieldnames=summary_fields(),
        exclude_names={
            "ranker_eval_summary.csv",
            "ranker_eval_summary_latest.csv",
            "ranker_eval_summary_all.csv",
        },
        required_substrings=("_runid",),
    )
    print(f"Updated latest files: {latest_detail_path}, {latest_summary_path}")
    print(f"Updated all files: {RESULT_DIR / 'ranker_eval_all.csv'}, {RESULT_DIR / 'ranker_eval_summary_all.csv'}")


def summarize_rows(rows: Iterable[Dict]) -> List[Dict]:
    rows = list(rows)
    grouped: Dict[tuple[str, str, str, int, str], List[Dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["size"], row["split"], int(row["top_k"]), row["model_tag"]), []).append(row)

    summary_rows = []
    for (method, size, split, top_k, model_tag), group in sorted(grouped.items()):
        out = {
            "method": method,
            "size": size,
            "split": split,
            "top_k": top_k,
            "model_tag": model_tag,
            "num_instances": len(group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out["valid_ratio"] = mean(1.0 if _truthy(row["is_valid_schedule"]) else 0.0 for row in group)
        out["cmax_check_pass_ratio"] = mean(1.0 if _truthy(row["cmax_check_passed"]) else 0.0 for row in group)
        summary_rows.append(out)
    return summary_rows


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--model_tag", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bc_model_path", default=None)
    parser.add_argument("--ranker_model_path", default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=DEFAULT_RULE_METHODS)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oracle_rollout_policy", choices=["fifo", "lookahead"], default="fifo")
    args = parser.parse_args()
    run_id = make_run_id(args.run_id)
    model_tag = infer_model_tag(args.methods, args.bc_model_path, args.ranker_model_path, args.model_tag)

    try:
        rows = evaluate_ranker(
            size=args.size,
            split=args.split,
            top_k=args.top_k,
            model_tag=model_tag,
            bc_model_path=args.bc_model_path,
            ranker_model_path=args.ranker_model_path,
            methods=args.methods,
            max_instances=args.max_instances,
            seed=args.seed,
            oracle_rollout_policy=args.oracle_rollout_policy,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    detail_path, summary_path = result_paths(args.size, args.split, args.top_k, model_tag, run_id, args.overwrite)
    detail_path = save_csv_no_overwrite(rows, detail_path, ROW_FIELDS, overwrite=True)
    summary_rows = summarize_rows(rows)
    summary_path = save_csv_no_overwrite(summary_rows, summary_path, summary_fields(), overwrite=True)
    update_latest_and_all(rows)
    print(f"Saved {len(rows)} rows to {detail_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
