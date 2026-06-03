"""Evaluate Stage F Conservative Rule-Selector PPO."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

import torch

from candidate_generator import fifo_ranked
from instance_manager import SIZES, SPLITS, ensure_fixed_dataset, load_fixed_instances
from mlp_models import load_checkpoint
from rule_selector_env import RuleSelectorEnv
from schedule_validator import VALIDATION_FIELDS, validate_schedule
from src.baselines.heuristics import choose_split_num
from src.envs.rolling_scheduling_env import RollingSchedulingEnv
from src.evaluation.metrics import compute_metrics
from stage_c_utils import best_candidate_index, extract_candidate_features, hybrid_candidates, oracle_cmax_per_candidate
from train_rule_selector_ppo import RuleActorCritic, select_action
from utils.experiment_io import make_result_path, make_run_id, save_csv_no_overwrite, update_latest_file, write_csv, progress_iter


METHODS = ["fifo", "hybrid_topk_fifo_select", "mlp_ranker_soft_ce", "rule_selector_ppo", "oracle_debug"]
RESULT_DIR = Path("data/results/stage_F")
METRICS = [
    "Cmax_roll",
    "average_completion_time",
    "average_waiting_time",
    "machine_utilization",
    "load_balance_std",
    "split_task_ratio",
    "total_split_count",
    "inference_time",
    "valid_ratio",
    "cmax_check_pass_ratio",
    "rule_fifo_ratio",
    "rule_lookahead_ratio",
    "rule_greedy_ratio",
    "rule_minload_ratio",
    "rule_ranker_ratio",
]
ROW_FIELDS = [
    "method",
    "size",
    "split",
    "top_k",
    "run_id",
    "instance_id",
    *METRICS,
    "rule_distribution",
    *VALIDATION_FIELDS,
]


def _fifo_candidate(env, candidates: List[str]) -> str:
    fifo_order = {job_id: idx for idx, job_id in enumerate(fifo_ranked(env))}
    return min(candidates, key=lambda j: (fifo_order.get(j, 10**9), j))


def _ranker_select(model, env, candidates: List[str]) -> str:
    if model is None:
        return _fifo_candidate(env, candidates)
    features = torch.tensor([extract_candidate_features(env, candidates)], dtype=torch.float32)
    with torch.no_grad():
        scores = model(features)[0]
    return candidates[int(torch.argmax(scores).item())]


def _load_rule_selector_model(path: str | None, state_dim: int, action_dim: int):
    if not path:
        raise ValueError("rule_selector_ppo requires --checkpoint_path.")
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Rule-Selector PPO checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = RuleActorCritic(int(checkpoint.get("state_dim", state_dim)), int(checkpoint.get("action_dim", action_dim)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_method(instance, method: str, top_k: int, seed: int, ranker_model=None, ppo_model=None, args=None) -> Dict:
    rng = random.Random(seed)
    start = time.perf_counter()
    rule_distribution = {rule: 0.0 for rule in ["fifo", "lookahead", "greedy_ect", "minload", "mlp_ranker_soft_ce", "mlp_ranker_pairwise"]}

    if method == "rule_selector_ppo":
        env = RuleSelectorEnv(
            instance,
            top_k=top_k,
            mlp_soft_model_path=args.mlp_soft_model_path if args else None,
            mlp_pairwise_model_path=args.mlp_pairwise_model_path if args else None,
        )
        state = env.reset(instance)
        model = ppo_model
        if model is None:
            model = _load_rule_selector_model(args.checkpoint_path, len(state), env.action_dim)
        while not env.env.is_done():
            action, _, _ = select_action(model, state, env.action_mask(), torch.device("cpu"), greedy=True)
            state, _, _, _ = env.step(action)
        base_env = env.env
        rule_distribution.update(env.rule_distribution())
    else:
        env = RollingSchedulingEnv(instance)
        env.reset(instance)
        while not env.is_done():
            if method == "fifo":
                job_id = fifo_ranked(env)[0]
            else:
                candidates = hybrid_candidates(env, top_k)
                if method == "hybrid_topk_fifo_select":
                    job_id = _fifo_candidate(env, candidates)
                elif method == "mlp_ranker_soft_ce":
                    job_id = _ranker_select(ranker_model, env, candidates)
                elif method == "oracle_debug":
                    values = oracle_cmax_per_candidate(env, candidates, rollout_policy="fifo")
                    job_id = candidates[best_candidate_index(values)]
                else:
                    job_id = rng.choice(candidates)
            env.step((job_id, choose_split_num(env, job_id)))
        base_env = env

    metrics = compute_metrics(base_env)
    validation = validate_schedule(base_env, instance)
    metrics["inference_time"] = time.perf_counter() - start
    metrics["valid_ratio"] = 1.0 if validation["is_valid_schedule"] else 0.0
    metrics["cmax_check_pass_ratio"] = 1.0 if validation["cmax_check_passed"] else 0.0
    metrics["rule_fifo_ratio"] = rule_distribution.get("fifo", 0.0)
    metrics["rule_lookahead_ratio"] = rule_distribution.get("lookahead", 0.0)
    metrics["rule_greedy_ratio"] = rule_distribution.get("greedy_ect", 0.0)
    metrics["rule_minload_ratio"] = rule_distribution.get("minload", 0.0)
    metrics["rule_ranker_ratio"] = rule_distribution.get("mlp_ranker_soft_ce", 0.0) + rule_distribution.get("mlp_ranker_pairwise", 0.0)
    metrics["rule_distribution"] = json.dumps(rule_distribution, sort_keys=True)
    metrics.update(validation)
    return metrics


def evaluate(args) -> List[Dict]:
    ensure_fixed_dataset([args.size], [args.split])
    instances = load_fixed_instances(args.size, args.split)
    if args.max_instances is not None:
        instances = instances[: max(1, args.max_instances)]

    ranker_model = load_checkpoint(args.mlp_soft_model_path) if args.mlp_soft_model_path and Path(args.mlp_soft_model_path).exists() else None
    ppo_model = None
    if "rule_selector_ppo" in args.methods and args.checkpoint_path:
        probe = RuleSelectorEnv(instances[0], top_k=args.top_k, mlp_soft_model_path=args.mlp_soft_model_path)
        ppo_model = _load_rule_selector_model(args.checkpoint_path, len(probe.reset(instances[0])), probe.action_dim)

    rows = []
    for method in args.methods:
        for instance in progress_iter(instances, desc=f"eval-stage-F {args.size}/{args.split} {method}", total=len(instances)):
            metrics = run_method(
                instance,
                method,
                top_k=args.top_k,
                seed=args.seed,
                ranker_model=ranker_model,
                ppo_model=ppo_model,
                args=args,
            )
            rows.append(
                {
                    "method": method,
                    "size": args.size,
                    "split": args.split,
                    "top_k": args.top_k,
                    "run_id": args.run_id,
                    "instance_id": getattr(instance, "instance_id", getattr(instance, "name", "")),
                    **metrics,
                }
            )
    return rows


def summarize_rows(rows: Iterable[Dict]) -> List[Dict]:
    rows = list(rows)
    grouped: Dict[tuple[str, str, str, int, str], List[Dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["size"], row["split"], int(row["top_k"]), row["run_id"]), []).append(row)
    summary = []
    for (method, size, split, top_k, run_id), group in sorted(grouped.items()):
        out = {"method": method, "size": size, "split": split, "top_k": top_k, "run_id": run_id, "num_instances": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summary.append(out)
    return summary


def summary_fields() -> List[str]:
    fields = ["method", "size", "split", "top_k", "run_id", "num_instances"]
    for metric in METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    return fields


def update_latest_all(rows: List[Dict]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    update_latest_file(rows, RESULT_DIR / "rule_selector_ppo_eval_latest.csv", ROW_FIELDS)
    summary = summarize_rows(rows)
    update_latest_file(summary, RESULT_DIR / "rule_selector_ppo_eval_summary_latest.csv", summary_fields())

    detail_all_path = RESULT_DIR / "rule_selector_ppo_eval_all.csv"
    summary_all_path = RESULT_DIR / "rule_selector_ppo_eval_summary_all.csv"
    detail_rows = []
    for path in sorted(RESULT_DIR.glob("rule_selector_ppo_eval_*_topk*_runid*.csv")):
        if path.name.endswith("_latest.csv") or "summary" in path.name:
            continue
        with path.open("r", newline="") as f:
            detail_rows.extend(csv.DictReader(f))
    write_csv(detail_rows, detail_all_path, ROW_FIELDS)

    summary_rows = []
    for path in sorted(RESULT_DIR.glob("rule_selector_ppo_eval_summary_*_topk*_runid*.csv")):
        if path.name.endswith("_latest.csv"):
            continue
        with path.open("r", newline="") as f:
            summary_rows.extend(csv.DictReader(f))
    write_csv(summary_rows, summary_all_path, summary_fields())
    write_csv(clean_stage_f_summary_rows(summary_rows), RESULT_DIR / "rule_selector_ppo_eval_summary_clean.csv", summary_fields())


def clean_stage_f_summary_rows(rows: Iterable[Dict]) -> List[Dict]:
    chosen: Dict[tuple[str, str, str, str], Dict] = {}
    for row in rows:
        key = (row["size"], row["split"], str(row["top_k"]), row["method"])
        current = chosen.get(key)
        if current is None or str(row.get("run_id", "")) >= str(current.get("run_id", "")):
            chosen[key] = row
    return [chosen[key] for key in sorted(chosen)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--mlp_soft_model_path", default=None)
    parser.add_argument("--mlp_pairwise_model_path", default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.run_id = make_run_id(args.run_id)

    rows = evaluate(args)
    tokens = [args.size, args.split, f"topk{args.top_k}", f"runid{args.run_id}"]
    detail_path = make_result_path(RESULT_DIR, "rule_selector_ppo_eval", tokens, None, overwrite=args.overwrite)
    summary_path = make_result_path(RESULT_DIR, "rule_selector_ppo_eval_summary", tokens, None, overwrite=args.overwrite)
    save_csv_no_overwrite(rows, detail_path, ROW_FIELDS, overwrite=True)
    save_csv_no_overwrite(summarize_rows(rows), summary_path, summary_fields(), overwrite=True)
    update_latest_all(rows)
    print(f"Saved Stage F eval rows to {detail_path}")
    print(f"Saved Stage F eval summary to {summary_path}")


if __name__ == "__main__":
    main()
