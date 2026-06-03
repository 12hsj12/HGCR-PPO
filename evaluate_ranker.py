"""Evaluate Stage C MLP-BC and MLP-Ranker policies on HybridTopK candidates."""

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
    sanitize_token,
    save_csv_no_overwrite,
    update_latest_file,
    progress_iter,
    write_csv,
)


RESULT_DIR = Path("data/results/stage_C")
STAGE_E_RESULT_DIR = Path("data/results/stage_E")
METHODS = [
    "hybrid_topk_random",
    "hybrid_topk_fifo_select",
    "mlp_bc",
    "mlp_ranker",
    "mlp_ranker_safe",
    "improvement_aware_ranker",
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
    "safe_fallback_count",
    "safe_ranker_override_count",
    "safe_fallback_ratio",
    "safe_ranker_override_ratio",
    "ranker_margin_mean",
    "ranker_margin_std",
    "improve_gate_mean",
    "improve_gate_std",
    "improve_override_count",
    "improve_fallback_count",
    "improve_override_ratio",
    "improve_fallback_ratio",
]
SAFE_METRICS = [
    "safe_fallback_count",
    "safe_ranker_override_count",
    "safe_fallback_ratio",
    "safe_ranker_override_ratio",
    "ranker_margin_mean",
    "ranker_margin_std",
]
IMPROVE_METRICS = [
    "improve_gate_mean",
    "improve_gate_std",
    "improve_override_count",
    "improve_fallback_count",
    "improve_override_ratio",
    "improve_fallback_ratio",
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


def _model_select(model, env, candidates: List[str]) -> str:
    scores = _model_scores(model, env, candidates)
    return candidates[int(max(range(len(scores)), key=lambda idx: scores[idx]))]


def _model_scores(model, env, candidates: List[str]) -> List[float]:
    import torch

    features = torch.tensor([extract_candidate_features(env, candidates)], dtype=torch.float32)
    with torch.no_grad():
        scores = model(features)[0]
    return [float(score) for score in scores.tolist()]


def _model_scores_and_gate(model, env, candidates: List[str]) -> tuple[List[float], float]:
    import torch

    if not hasattr(model, "forward_with_gate"):
        raise ValueError(
            "improvement_aware_ranker requires a checkpoint trained with --loss_type improvement_aware."
        )
    features = torch.tensor([extract_candidate_features(env, candidates)], dtype=torch.float32)
    mask = torch.ones(1, len(candidates), dtype=torch.bool)
    with torch.no_grad():
        scores, gate_logits = model.forward_with_gate(features, mask)
        p_improve = torch.sigmoid(gate_logits)[0]
    return [float(score) for score in scores[0].tolist()], float(p_improve)


def _fifo_candidate(env, candidates: List[str]) -> str:
    fifo_order = {job_id: idx for idx, job_id in enumerate(fifo_ranked(env))}
    return min(candidates, key=lambda j: (fifo_order.get(j, 10**9), j))


def _ranker_safe_select(
    model,
    env,
    candidates: List[str],
    ranker_margin_threshold: float,
) -> tuple[str, Dict[str, float]]:
    if model is None:
        raise ValueError(
            "mlp_ranker_safe requires --ranker_model_path. Please train MLP-Ranker first or remove "
            "mlp_ranker_safe from --methods."
        )
    scores = _model_scores(model, env, candidates)
    score_by_job = {job_id: scores[idx] for idx, job_id in enumerate(candidates)}
    fifo_job = _fifo_candidate(env, candidates)
    ranker_job = candidates[int(max(range(len(scores)), key=lambda idx: scores[idx]))]
    margin = float(score_by_job[ranker_job] - score_by_job[fifo_job])

    if ranker_job == fifo_job or margin < ranker_margin_threshold:
        return fifo_job, {"fallback": 1.0, "override": 0.0, "margin": margin}
    return ranker_job, {"fallback": 0.0, "override": 1.0, "margin": margin}


def _improvement_aware_select(
    model,
    env,
    candidates: List[str],
    improve_threshold: float,
) -> tuple[str, Dict[str, float]]:
    if model is None:
        raise ValueError(
            "improvement_aware_ranker requires --ranker_model_path. Please train with "
            "--loss_type improvement_aware first or remove improvement_aware_ranker from --methods."
        )
    scores, p_improve = _model_scores_and_gate(model, env, candidates)
    ranker_job = candidates[int(max(range(len(scores)), key=lambda idx: scores[idx]))]
    fifo_job = _fifo_candidate(env, candidates)
    if p_improve >= improve_threshold:
        return ranker_job, {"gate": p_improve, "override": 1.0, "fallback": 0.0}
    return fifo_job, {"gate": p_improve, "override": 0.0, "fallback": 1.0}


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
        return _fifo_candidate(env, candidates)
    if method in {"mlp_bc", "mlp_ranker", "mlp_ranker_safe", "improvement_aware_ranker"}:
        if model is None:
            if method == "mlp_bc":
                raise ValueError(
                    "mlp_bc requires --bc_model_path. Please train MLP-BC first or remove mlp_bc from --methods."
                )
            raise ValueError(
                f"{method} requires --ranker_model_path. Please train MLP-Ranker first or remove {method} from --methods."
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
    ranker_margin_threshold: float,
    improve_threshold: float,
    model=None,
) -> Dict:
    rng = random.Random(seed)
    env = RollingSchedulingEnv(instance)
    env.reset(instance)
    oracle_matches = []
    fifo_matches = []
    safe_fallbacks = []
    safe_overrides = []
    ranker_margins = []
    improve_gates = []
    improve_overrides = []
    improve_fallbacks = []
    start = time.perf_counter()

    while not env.is_done():
        candidates = hybrid_candidates(env, top_k)
        oracle_cmax = oracle_cmax_per_candidate(env, candidates, rollout_policy=oracle_rollout_policy)
        oracle_job = candidates[best_candidate_index(oracle_cmax)]
        fifo_job = fifo_first(env)
        if method == "mlp_ranker_safe":
            job_id, safe_diag = _ranker_safe_select(model, env, candidates, ranker_margin_threshold)
            safe_fallbacks.append(safe_diag["fallback"])
            safe_overrides.append(safe_diag["override"])
            ranker_margins.append(safe_diag["margin"])
        elif method == "improvement_aware_ranker":
            job_id, improve_diag = _improvement_aware_select(model, env, candidates, improve_threshold)
            improve_gates.append(improve_diag["gate"])
            improve_overrides.append(improve_diag["override"])
            improve_fallbacks.append(improve_diag["fallback"])
        else:
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
    metrics["safe_fallback_count"] = sum(safe_fallbacks)
    metrics["safe_ranker_override_count"] = sum(safe_overrides)
    metrics["safe_fallback_ratio"] = mean(safe_fallbacks) if safe_fallbacks else 0.0
    metrics["safe_ranker_override_ratio"] = mean(safe_overrides) if safe_overrides else 0.0
    metrics["ranker_margin_mean"] = mean(ranker_margins) if ranker_margins else 0.0
    metrics["ranker_margin_std"] = pstdev(ranker_margins) if len(ranker_margins) > 1 else 0.0
    metrics["improve_gate_mean"] = mean(improve_gates) if improve_gates else 0.0
    metrics["improve_gate_std"] = pstdev(improve_gates) if len(improve_gates) > 1 else 0.0
    metrics["improve_override_count"] = sum(improve_overrides)
    metrics["improve_fallback_count"] = sum(improve_fallbacks)
    metrics["improve_override_ratio"] = mean(improve_overrides) if improve_overrides else 0.0
    metrics["improve_fallback_ratio"] = mean(improve_fallbacks) if improve_fallbacks else 0.0
    metrics.update(validate_schedule(env, instance))
    return metrics


def evaluate_ranker(
    size: str,
    split: str,
    top_k: int,
    model_tag: str,
    run_id: str,
    bc_model_path: str | None,
    ranker_model_path: str | None,
    methods: List[str],
    max_instances: int | None,
    seed: int,
    oracle_rollout_policy: str,
    ranker_margin_threshold: float,
    improve_threshold: float,
) -> List[Dict]:
    ensure_fixed_dataset([size], [split])
    instances = load_fixed_instances(size, split)
    if max_instances is not None:
        instances = instances[: max(0, max_instances)]

    models = _load_models_for_methods(methods, bc_model_path, ranker_model_path, size, top_k)
    rows: List[Dict] = []
    for method in methods:
        desc = f"eval {size}/{split} topk{top_k} {method}"
        for instance in progress_iter(instances, desc=desc, total=len(instances)):
            model = models.get(method)
            metrics = run_method(
                instance,
                method,
                top_k,
                seed,
                oracle_rollout_policy,
                ranker_margin_threshold=ranker_margin_threshold,
                improve_threshold=improve_threshold,
                model=model,
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


def _load_models_for_methods(
    methods: List[str],
    bc_model_path: str | None,
    ranker_model_path: str | None,
    size: str,
    top_k: int,
) -> Dict[str, object]:
    from mlp_models import load_checkpoint

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
    ranker_methods = {"mlp_ranker", "mlp_ranker_safe", "improvement_aware_ranker"}
    if any(method in ranker_methods for method in methods):
        if not ranker_model_path:
            raise ValueError(
                "ranker methods require --ranker_model_path. Please train the requested ranker first "
                "or remove ranker methods from --methods."
            )
        ranker_path = Path(ranker_model_path)
        if not ranker_path.exists():
            raise FileNotFoundError(_missing_ranker_checkpoint_message(ranker_path, size, top_k))
        ranker_model = load_checkpoint(ranker_path)
        models["mlp_ranker"] = ranker_model
        models["mlp_ranker_safe"] = ranker_model
        models["improvement_aware_ranker"] = ranker_model
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
        f"checkpoints/stage_C/mlp_ranker/{size}_topk{top_k}_{{loss_type}}_latest/best.pt\n"
        "For improvement-aware ranker, expected Stage E paths are:\n"
        f"checkpoints/stage_E/improvement_ranker/{size}_topk{top_k}_eps{{epsilon}}_runid{{run_id}}/best.pt\n"
        f"checkpoints/stage_E/improvement_ranker/{size}_topk{top_k}_eps{{epsilon}}_latest/best.pt"
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
    fieldnames = ["method", "size", "split", "top_k", "model_tag", "run_id", "num_instances"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    fieldnames.extend(["valid_ratio", "cmax_check_pass_ratio"])
    return fieldnames


def _is_stage_e_run(methods: List[str]) -> bool:
    return "improvement_aware_ranker" in methods


def _result_config(methods: List[str]) -> tuple[Path, str, str]:
    if _is_stage_e_run(methods):
        return STAGE_E_RESULT_DIR, "improvement_ranker_eval", "improvement_ranker_eval_summary"
    return RESULT_DIR, "ranker_eval", "ranker_eval_summary"


def result_paths(
    size: str,
    split: str,
    top_k: int,
    model_tag: str,
    run_id: str,
    overwrite: bool,
    methods: List[str],
) -> tuple[Path, Path]:
    tokens = [size, split, f"topk{top_k}", model_tag, f"runid{run_id}"]
    result_dir, detail_prefix, summary_prefix = _result_config(methods)
    return (
        make_result_path(result_dir, detail_prefix, tokens, run_id="", overwrite=overwrite),
        make_result_path(result_dir, summary_prefix, tokens, run_id="", overwrite=overwrite),
    )


def infer_model_tag(methods: List[str], bc_model_path: str | None, ranker_model_path: str | None, model_tag: str | None) -> str:
    if model_tag:
        return sanitize_token(model_tag)
    model_methods = {"mlp_bc", "mlp_ranker", "mlp_ranker_safe", "improvement_aware_ranker"}
    if not any(method in model_methods for method in methods):
        return "rules_only"
    ranker_path = str(ranker_model_path or "").lower()
    if "soft_ce" in ranker_path:
        return "soft_ce"
    if "pairwise" in ranker_path:
        return "pairwise"
    if "improvement_ranker" in ranker_path or "improvement_aware" in ranker_path:
        return "improvement_aware"
    if "mlp_bc" in methods and not ranker_model_path:
        return "bc_only"
    return "manual"


def update_latest_and_all(rows: List[Dict], methods: List[str]) -> None:
    result_dir, detail_prefix, summary_prefix = _result_config(methods)
    latest_detail_path = result_dir / f"{detail_prefix}_latest.csv"
    latest_summary_path = result_dir / f"{summary_prefix}_latest.csv"
    update_latest_file(rows, latest_detail_path, ROW_FIELDS)
    summary_rows = summarize_rows(rows)
    update_latest_file(summary_rows, latest_summary_path, summary_fields())
    rebuild_ranker_all_outputs(methods)
    print(f"Updated latest files: {latest_detail_path}, {latest_summary_path}")
    print(
        "Updated all files: "
        f"{result_dir / f'{detail_prefix}_all.csv'}, "
        f"{result_dir / f'{summary_prefix}_all.csv'}, "
        f"{result_dir / f'{summary_prefix}_clean.csv'}"
    )


def rebuild_ranker_all_outputs(methods: List[str] | None = None) -> tuple[Path, Path]:
    methods = methods or []
    result_dir, detail_prefix, summary_prefix = _result_config(methods)
    detail_path = result_dir / f"{detail_prefix}_all.csv"
    summary_path = result_dir / f"{summary_prefix}_all.csv"
    clean_summary_path = result_dir / f"{summary_prefix}_clean.csv"
    detail_rows = collect_ranker_csv_rows(
        pattern=f"{detail_prefix}_*_topk*.csv",
        output_fields=ROW_FIELDS,
        required_fields=["method", "size", "split", "top_k", "instance_id"],
        filename_prefix=detail_prefix,
        result_dir=result_dir,
        exclude_names={
            f"{detail_prefix}.csv",
            f"{detail_prefix}_latest.csv",
            f"{detail_prefix}_all.csv",
            f"{summary_prefix}.csv",
            f"{summary_prefix}_latest.csv",
            f"{summary_prefix}_all.csv",
        },
        exclude_prefixes=(f"{summary_prefix}_",),
    )
    summary_rows = collect_ranker_csv_rows(
        pattern=f"{summary_prefix}_*_topk*.csv",
        output_fields=summary_fields(),
        required_fields=["method", "size", "split", "top_k", "num_instances"],
        filename_prefix=summary_prefix,
        result_dir=result_dir,
        exclude_names={
            f"{summary_prefix}.csv",
            f"{summary_prefix}_latest.csv",
            f"{summary_prefix}_all.csv",
        },
    )
    write_csv(detail_rows, detail_path, ROW_FIELDS)
    write_csv(summary_rows, summary_path, summary_fields())
    write_csv(clean_summary_rows(summary_rows), clean_summary_path, summary_fields())
    return detail_path, summary_path


def clean_summary_rows(rows: Iterable[Dict]) -> List[Dict]:
    chosen: Dict[tuple[str, str, str, str, str], Dict] = {}
    for row in rows:
        key = (row["size"], row["split"], str(row["top_k"]), row["model_tag"], row["method"])
        current = chosen.get(key)
        if current is None or _summary_row_rank(row) >= _summary_row_rank(current):
            chosen[key] = row
    return [chosen[key] for key in sorted(chosen)]


def _summary_row_rank(row: Dict) -> tuple[int, str]:
    run_id = str(row.get("run_id") or "legacy")
    return (0 if run_id == "legacy" else 1, run_id)


def collect_ranker_csv_rows(
    pattern: str,
    output_fields: List[str],
    required_fields: List[str],
    filename_prefix: str,
    result_dir: Path,
    exclude_names: set[str],
    exclude_prefixes: tuple[str, ...] = (),
) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(result_dir.glob(pattern)):
        if path.name in exclude_names or any(path.name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if path.suffix.lower() != ".csv":
            continue
        metadata = parse_ranker_result_filename(path.name, filename_prefix)
        if metadata is None:
            print(f"Warning: skipped {path} because filename does not match the Stage C ranker schema.")
            continue
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if not _compatible_ranker_fields(fieldnames, output_fields, required_fields):
                print(
                    f"Warning: skipped {path} because CSV fields do not contain the required Stage C schema fields."
                )
                continue
            has_run_id = "run_id" in fieldnames
            for row in reader:
                normalized = {field: row.get(field, "") for field in output_fields}
                normalized["size"] = normalized.get("size") or metadata["size"]
                normalized["split"] = normalized.get("split") or metadata["split"]
                normalized["top_k"] = normalized.get("top_k") or metadata["top_k"]
                normalized["model_tag"] = normalized.get("model_tag") or metadata["model_tag"]
                if not has_run_id or not normalized.get("run_id"):
                    normalized["run_id"] = metadata["run_id"]
                rows.append(normalized)
    return rows


def _compatible_ranker_fields(fieldnames: List[str], output_fields: List[str], required_fields: List[str]) -> bool:
    field_set = set(fieldnames)
    output_set = set(output_fields)
    return set(required_fields).issubset(field_set) and field_set.issubset(output_set)


def parse_ranker_result_filename(filename: str, prefix: str) -> Dict[str, str] | None:
    stem = Path(filename).stem
    marker = f"{prefix}_"
    if not stem.startswith(marker):
        return None
    tail = stem[len(marker) :]
    parts = tail.split("_", 3)
    if len(parts) < 4:
        return None
    size, split, topk_token, tag_and_run = parts
    if size not in SIZES or split not in SPLITS or not topk_token.startswith("topk"):
        return None
    top_k = topk_token[4:]
    if not top_k.isdigit():
        return None
    if "_runid" in tag_and_run:
        model_tag, run_id = tag_and_run.rsplit("_runid", 1)
        run_id = run_id or "manual"
    else:
        model_tag = tag_and_run
        run_id = "legacy"
    return {"size": size, "split": split, "top_k": top_k, "model_tag": model_tag or "manual", "run_id": run_id}


def summarize_rows(rows: Iterable[Dict]) -> List[Dict]:
    rows = list(rows)
    grouped: Dict[tuple[str, str, str, int, str, str], List[Dict]] = {}
    for row in rows:
        grouped.setdefault(
            (row["method"], row["size"], row["split"], int(row["top_k"]), row["model_tag"], row["run_id"]),
            [],
        ).append(row)

    summary_rows = []
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
        summary_rows.append(out)
    return summary_rows


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild_all", action="store_true")
    parser.add_argument("--size", choices=SIZES, default=None)
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
    parser.add_argument("--ranker_margin_threshold", type=float, default=0.0)
    parser.add_argument("--improve_threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.rebuild_all:
        detail_path, summary_path = rebuild_ranker_all_outputs(args.methods)
        _, _, summary_prefix = _result_config(args.methods)
        print(f"Rebuilt all files: {detail_path}, {summary_path}, {detail_path.parent / f'{summary_prefix}_clean.csv'}")
        return
    if args.size is None:
        parser.error("--size is required unless --rebuild_all is used.")
    run_id = make_run_id(args.run_id)
    model_tag = infer_model_tag(args.methods, args.bc_model_path, args.ranker_model_path, args.model_tag)

    try:
        rows = evaluate_ranker(
            size=args.size,
            split=args.split,
            top_k=args.top_k,
            model_tag=model_tag,
            run_id=run_id,
            bc_model_path=args.bc_model_path,
            ranker_model_path=args.ranker_model_path,
            methods=args.methods,
            max_instances=args.max_instances,
            seed=args.seed,
            oracle_rollout_policy=args.oracle_rollout_policy,
            ranker_margin_threshold=args.ranker_margin_threshold,
            improve_threshold=args.improve_threshold,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    detail_path, summary_path = result_paths(
        args.size,
        args.split,
        args.top_k,
        model_tag,
        run_id,
        args.overwrite,
        args.methods,
    )
    detail_path = save_csv_no_overwrite(rows, detail_path, ROW_FIELDS, overwrite=True)
    summary_rows = summarize_rows(rows)
    summary_path = save_csv_no_overwrite(summary_rows, summary_path, summary_fields(), overwrite=True)
    update_latest_and_all(rows, args.methods)
    print(f"Saved {len(rows)} rows to {detail_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
