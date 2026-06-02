"""Evaluate Stage D GNN-Ranker against minimal graph-structure baselines."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

from candidate_generator import fifo_ranked
from instance_manager import SIZES, SPLITS, ensure_fixed_dataset, load_fixed_instances
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
    progress_iter,
    sanitize_token,
    save_csv_no_overwrite,
    update_latest_file,
    write_csv,
)


RESULT_DIR = Path("data/results/stage_D")
METHODS = [
    "hybrid_topk_random",
    "hybrid_topk_fifo_select",
    "mlp_ranker",
    "gnn_ranker",
    "oracle_debug",
]
DEFAULT_METHODS = ["hybrid_topk_random", "hybrid_topk_fifo_select", "gnn_ranker", "oracle_debug"]
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
    "run_id",
    "instance_id",
    *METRICS,
    *VALIDATION_FIELDS,
]


def _fifo_candidate(env, candidates: List[str]) -> str:
    fifo_order = {job_id: idx for idx, job_id in enumerate(fifo_ranked(env))}
    return min(candidates, key=lambda job_id: (fifo_order.get(job_id, 10**9), job_id))


def _mlp_scores(model, env, candidates: List[str]) -> List[float]:
    import torch

    features = torch.tensor([extract_candidate_features(env, candidates)], dtype=torch.float32)
    with torch.no_grad():
        scores = model(features)[0]
    return [float(score) for score in scores.tolist()]


def _gnn_scores(model, env, candidates: List[str], device: str) -> List[float]:
    import torch
    from gnn_graph_builder import build_graph_from_env
    from gnn_ranker_models import graph_to_torch

    graph = graph_to_torch(build_graph_from_env(env, candidates), device=device)
    with torch.no_grad():
        scores = model.forward_graph(graph)
    return [float(score) for score in scores.detach().cpu().tolist()]


def _select_job(
    method: str,
    env,
    candidates: List[str],
    rng: random.Random,
    oracle_rollout_policy: str,
    mlp_model=None,
    gnn_model=None,
    gnn_device: str = "cpu",
    oracle_cmax: List[float] | None = None,
) -> str:
    if method == "hybrid_topk_random":
        return rng.choice(candidates)
    if method == "hybrid_topk_fifo_select":
        return _fifo_candidate(env, candidates)
    if method == "mlp_ranker":
        if mlp_model is None:
            raise ValueError("mlp_ranker requires --mlp_model_path.")
        scores = _mlp_scores(mlp_model, env, candidates)
        return candidates[int(max(range(len(scores)), key=lambda idx: scores[idx]))]
    if method == "gnn_ranker":
        if gnn_model is None:
            raise ValueError("gnn_ranker requires --gnn_model_path.")
        scores = _gnn_scores(gnn_model, env, candidates, gnn_device)
        return candidates[int(max(range(len(scores)), key=lambda idx: scores[idx]))]
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
    mlp_model=None,
    gnn_model=None,
    gnn_device: str = "cpu",
) -> Dict:
    rng = random.Random(seed)
    env = RollingSchedulingEnv(instance)
    env.reset(instance)
    oracle_matches = []
    fifo_matches = []
    started = time.perf_counter()

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
            mlp_model=mlp_model,
            gnn_model=gnn_model,
            gnn_device=gnn_device,
            oracle_cmax=oracle_cmax,
        )
        oracle_matches.append(1.0 if job_id == oracle_job else 0.0)
        fifo_matches.append(1.0 if fifo_job is not None and job_id == fifo_job else 0.0)
        env.step((job_id, choose_split_num(env, job_id)))

    metrics = compute_metrics(env)
    metrics["inference_time"] = time.perf_counter() - started
    metrics["action_match_oracle_ratio"] = mean(oracle_matches) if oracle_matches else 0.0
    metrics["action_match_fifo_ratio"] = mean(fifo_matches) if fifo_matches else 0.0
    metrics.update(validate_schedule(env, instance))
    return metrics


def evaluate_gnn_ranker(
    size: str,
    split: str,
    top_k: int,
    gnn_model_path: str | None,
    mlp_model_path: str | None,
    model_tag: str,
    run_id: str,
    methods: List[str],
    max_instances: int | None,
    seed: int,
    oracle_rollout_policy: str,
    device: str,
) -> List[Dict]:
    ensure_fixed_dataset([size], [split])
    instances = load_fixed_instances(size, split)
    if max_instances is not None:
        instances = instances[: max(0, max_instances)]
    models = _load_models(methods, gnn_model_path, mlp_model_path, device)

    rows: List[Dict] = []
    for method in methods:
        desc = f"eval-gnn {size}/{split} topk{top_k} {method}"
        for instance in progress_iter(instances, desc=desc, total=len(instances)):
            metrics = run_method(
                instance,
                method,
                top_k,
                seed,
                oracle_rollout_policy,
                mlp_model=models.get("mlp_ranker"),
                gnn_model=models.get("gnn_ranker"),
                gnn_device=device,
            )
            rows.append(
                {
                    "method": method,
                    "size": size,
                    "split": split,
                    "top_k": top_k,
                    "model_tag": model_tag,
                    "run_id": run_id,
                    "instance_id": getattr(instance, "instance_id", instance.name),
                    **metrics,
                }
            )
    return rows


def _load_models(methods: List[str], gnn_model_path: str | None, mlp_model_path: str | None, device: str) -> Dict[str, object]:
    models: Dict[str, object] = {}
    if "mlp_ranker" in methods:
        if not mlp_model_path:
            raise ValueError("mlp_ranker requires --mlp_model_path.")
        from mlp_models import load_checkpoint as load_mlp_checkpoint

        path = Path(mlp_model_path)
        if not path.exists():
            raise FileNotFoundError(f"MLP-Ranker checkpoint not found: {path}")
        models["mlp_ranker"] = load_mlp_checkpoint(path, device=device)
    if "gnn_ranker" in methods:
        if not gnn_model_path:
            raise ValueError("gnn_ranker requires --gnn_model_path.")
        from gnn_ranker_models import load_checkpoint as load_gnn_checkpoint

        path = Path(gnn_model_path)
        if not path.exists():
            raise FileNotFoundError(f"GNN-Ranker checkpoint not found: {path}")
        models["gnn_ranker"] = load_gnn_checkpoint(path, device=device)
    return models


def result_paths(size: str, split: str, top_k: int, model_tag: str, run_id: str, overwrite: bool) -> tuple[Path, Path]:
    tokens = [size, split, f"topk{top_k}", model_tag, f"runid{run_id}"]
    return (
        make_result_path(RESULT_DIR, "gnn_ranker_eval", tokens, run_id=None, overwrite=overwrite),
        make_result_path(RESULT_DIR, "gnn_ranker_eval_summary", tokens, run_id=None, overwrite=overwrite),
    )


def summary_fields() -> List[str]:
    fields = ["method", "size", "split", "top_k", "model_tag", "run_id", "num_instances"]
    for metric in METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    fields.extend(["valid_ratio", "cmax_check_pass_ratio"])
    return fields


def summarize_rows(rows: Iterable[Dict]) -> List[Dict]:
    rows = list(rows)
    grouped: Dict[tuple[str, str, str, int, str, str], List[Dict]] = {}
    for row in rows:
        grouped.setdefault(
            (row["method"], row["size"], row["split"], int(row["top_k"]), row["model_tag"], row["run_id"]),
            [],
        ).append(row)
    out_rows = []
    for (method, size, split, top_k, model_tag, run_id), group in sorted(grouped.items()):
        out = {
            "method": method,
            "size": size,
            "split": split,
            "top_k": top_k,
            "model_tag": model_tag,
            "run_id": run_id,
            "num_instances": len(group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out["valid_ratio"] = mean(1.0 if _truthy(row["is_valid_schedule"]) else 0.0 for row in group)
        out["cmax_check_pass_ratio"] = mean(1.0 if _truthy(row["cmax_check_passed"]) else 0.0 for row in group)
        out_rows.append(out)
    return out_rows


def update_latest_and_all(rows: List[Dict]) -> None:
    latest_detail = RESULT_DIR / "gnn_ranker_eval_latest.csv"
    latest_summary = RESULT_DIR / "gnn_ranker_eval_summary_latest.csv"
    update_latest_file(rows, latest_detail, ROW_FIELDS)
    update_latest_file(summarize_rows(rows), latest_summary, summary_fields())
    rebuild_all_outputs()
    print(f"Updated latest files: {latest_detail}, {latest_summary}")


def rebuild_all_outputs() -> tuple[Path, Path]:
    detail_rows = _collect_csv_rows(
        pattern="gnn_ranker_eval_*_topk*.csv",
        fields=ROW_FIELDS,
        exclude_names={"gnn_ranker_eval_latest.csv", "gnn_ranker_eval_all.csv"},
        exclude_prefixes=("gnn_ranker_eval_summary_",),
    )
    summary_rows = _collect_csv_rows(
        pattern="gnn_ranker_eval_summary_*_topk*.csv",
        fields=summary_fields(),
        exclude_names={"gnn_ranker_eval_summary_latest.csv", "gnn_ranker_eval_summary_all.csv"},
    )
    detail_path = write_csv(detail_rows, RESULT_DIR / "gnn_ranker_eval_all.csv", ROW_FIELDS)
    summary_path = write_csv(summary_rows, RESULT_DIR / "gnn_ranker_eval_summary_all.csv", summary_fields())
    return detail_path, summary_path


def _collect_csv_rows(pattern: str, fields: List[str], exclude_names: set[str], exclude_prefixes=()) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(RESULT_DIR.glob(pattern)):
        if path.name in exclude_names or any(path.name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if not set(reader.fieldnames or []).issubset(set(fields)):
                print(f"Warning: skipped {path} because CSV fields do not match Stage D schema.")
                continue
            rows.extend({field: row.get(field, "") for field in fields} for row in reader)
    return rows


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def infer_model_tag(model_tag: str | None, gnn_model_path: str | None, methods: List[str]) -> str:
    if model_tag:
        return sanitize_token(model_tag)
    if "gnn_ranker" in methods and gnn_model_path:
        path = str(gnn_model_path).lower()
        if "soft_ce" in path:
            return "gnn_soft_ce"
        return "gnn_manual"
    return "rules_only"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild_all", action="store_true")
    parser.add_argument("--size", choices=SIZES, default=None)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--gnn_model_path", default=None)
    parser.add_argument("--mlp_model_path", default=None)
    parser.add_argument("--model_tag", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oracle_rollout_policy", choices=["fifo", "lookahead"], default="fifo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.rebuild_all:
        detail_path, summary_path = rebuild_all_outputs()
        print(f"Rebuilt all files: {detail_path}, {summary_path}")
        return
    if args.size is None:
        parser.error("--size is required unless --rebuild_all is used.")

    run_id = make_run_id(args.run_id)
    model_tag = infer_model_tag(args.model_tag, args.gnn_model_path, args.methods)
    try:
        rows = evaluate_gnn_ranker(
            size=args.size,
            split=args.split,
            top_k=args.top_k,
            gnn_model_path=args.gnn_model_path,
            mlp_model_path=args.mlp_model_path,
            model_tag=model_tag,
            run_id=run_id,
            methods=args.methods,
            max_instances=args.max_instances,
            seed=args.seed,
            oracle_rollout_policy=args.oracle_rollout_policy,
            device=args.device,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    detail_path, summary_path = result_paths(args.size, args.split, args.top_k, model_tag, run_id, args.overwrite)
    detail_path = save_csv_no_overwrite(rows, detail_path, ROW_FIELDS, overwrite=True)
    summary_path = save_csv_no_overwrite(summarize_rows(rows), summary_path, summary_fields(), overwrite=True)
    update_latest_and_all(rows)
    print(f"Saved {len(rows)} rows to {detail_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
