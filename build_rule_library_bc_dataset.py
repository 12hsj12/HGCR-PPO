"""Build rule-level BC dataset for the Stage F experience-guided rule library."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

from candidate_generator import generate_candidates
from instance_manager import SIZES, SPLITS, ensure_fixed_dataset, load_fixed_instances
from src.baselines.heuristics import choose_split_num
from src.envs.rolling_scheduling_env import RollingSchedulingEnv
from src.rules.experience_rule_library import ExperienceRuleLibrary, RULE_NAMES
from utils.experiment_io import write_csv


PREVIEW_FIELDS = [
    "instance_id",
    "step_id",
    "label_rule_id",
    "label_rule_name",
    "chosen_job_id",
    "candidate_count",
]
LABEL_FIELDS = [
    "rule_name",
    "label_count",
    "label_ratio",
    "valid_count",
    "valid_ratio",
    "mean_proxy_score",
    "is_excluded_from_label",
    "eligible_label_count",
    "eligible_label_ratio",
]
ELIGIBLE_TIEBREAK = {
    "mlp_ranker_soft_ce": 0,
    "fifo": 1,
    "lookahead": 2,
    "greedy_ect": 3,
    "minload": 4,
}


def make_run_id(args) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label_tokens = {
        "one_step_cmax": "osc",
        "one_step_ect": "ose",
        "conservative_margin": "cm",
    }
    label_token = label_tokens.get(args.label_mode, args.label_mode.replace("_", ""))
    return f"RLBData_{args.size}_{args.split}_k{args.top_k}_{label_token}_{stamp}_{uuid.uuid4().hex[:8]}"


def output_paths(args, run_id: str) -> Dict[str, Path]:
    run_dir = Path(args.output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return {
        "run_dir": run_dir,
        "dataset": run_dir / "dataset.pt",
        "preview": run_dir / "preview.csv",
        "label_stats": run_dir / "label_stats.csv",
        "manifest": run_dir / "manifest.json",
    }


def excluded_label_rules(args) -> set[str]:
    return set(args.exclude_label_rules or [])


def eligible_recommendations(args, recommendations):
    excluded = excluded_label_rules(args)
    return [rec for rec in recommendations if rec.is_valid and rec.rule_name not in excluded]


def select_by_proxy_with_eligible_tiebreak(recommendations):
    return min(recommendations, key=lambda rec: (rec.score_or_proxy, ELIGIBLE_TIEBREAK.get(rec.rule_name, 999), rec.rule_id))


def select_label(args, library: ExperienceRuleLibrary, recommendations):
    eligible = eligible_recommendations(args, recommendations)
    if not eligible:
        raise RuntimeError("No eligible valid rule remains after --exclude_label_rules filtering.")

    if args.label_mode in {"one_step_cmax", "one_step_ect"}:
        return select_by_proxy_with_eligible_tiebreak(eligible)

    if args.label_mode != "conservative_margin":
        raise ValueError(f"Unsupported label_mode: {args.label_mode}")

    baseline = next((rec for rec in recommendations if rec.rule_name == args.baseline_rule), None)
    if baseline is None or not baseline.is_valid:
        raise RuntimeError(
            f"conservative_margin requires a valid baseline_rule={args.baseline_rule!r}; "
            "check ranker checkpoint and HybridTopK candidates."
        )
    if baseline.rule_name in excluded_label_rules(args):
        raise RuntimeError(f"baseline_rule={baseline.rule_name!r} cannot be excluded from labels.")

    other_valid = [rec for rec in eligible if rec.rule_name != args.baseline_rule]
    if not other_valid:
        return baseline

    best_other = select_by_proxy_with_eligible_tiebreak(other_valid)
    if best_other.score_or_proxy < baseline.score_or_proxy - float(args.label_margin):
        return best_other
    return baseline


def build_dataset(args) -> Dict[str, Path]:
    run_id = make_run_id(args)
    paths = output_paths(args, run_id)
    ensure_fixed_dataset([args.size], [args.split])
    instances = load_fixed_instances(args.size, args.split)
    if args.max_instances is not None:
        instances = instances[: max(1, args.max_instances)]
    if args.smoke_test:
        instances = instances[:1]

    library = ExperienceRuleLibrary(ranker_ckpt=args.ranker_ckpt, device=args.device)
    records: List[Dict] = []
    preview_rows = []
    label_counts = Counter()
    valid_counts = Counter()
    proxy_sums = defaultdict(float)
    eligible_counts = Counter()

    for instance in instances:
        env = RollingSchedulingEnv(instance)
        env.reset(instance)
        step_id = 0
        while not env.is_done():
            candidates = generate_candidates(env, candidate_mode="hybrid_topk", top_k=args.top_k, fallback_to_all=True)
            recommendations = library.recommend(env, candidates)
            label = select_label(args, library, recommendations)
            state_features = library.state_features(env, candidates, recommendations)
            candidate_features = []
            try:
                from stage_c_utils import extract_candidate_features

                candidate_features = extract_candidate_features(env, candidates)
            except Exception:
                candidate_features = []

            records.append(
                {
                    "instance_id": getattr(instance, "instance_id", getattr(instance, "name", "")),
                    "step_id": step_id,
                    "state_features": state_features,
                    "candidate_features": candidate_features,
                    "rule_job_ids": [rec.job_id for rec in recommendations],
                    "rule_candidate_indices": [rec.candidate_index for rec in recommendations],
                    "rule_valid_mask": [bool(rec.is_valid) for rec in recommendations],
                    "rule_proxy_scores": [float(rec.score_or_proxy) for rec in recommendations],
                    "label_rule_id": int(label.rule_id),
                    "label_rule_name": label.rule_name,
                    "chosen_job_id": label.job_id,
                    "candidate_count": len(candidates),
                }
            )
            preview_rows.append(
                {
                    "instance_id": getattr(instance, "instance_id", getattr(instance, "name", "")),
                    "step_id": step_id,
                    "label_rule_id": label.rule_id,
                    "label_rule_name": label.rule_name,
                    "chosen_job_id": label.job_id,
                    "candidate_count": len(candidates),
                }
            )
            label_counts[label.rule_name] += 1
            for rec in recommendations:
                if rec.is_valid:
                    valid_counts[rec.rule_name] += 1
                    proxy_sums[rec.rule_name] += float(rec.score_or_proxy)
                    if rec.rule_name not in excluded_label_rules(args):
                        eligible_counts[rec.rule_name] += 1
            env.step((str(label.job_id), choose_split_num(env, str(label.job_id))))
            step_id += 1

    if paths["dataset"].exists():
        raise FileExistsError(paths["dataset"])
    torch.save({"records": records, "rule_names": RULE_NAMES, "run_id": run_id, "args": vars(args)}, paths["dataset"])
    write_csv(preview_rows, paths["preview"], PREVIEW_FIELDS)
    total_labels = max(1, sum(label_counts.values()))
    total_states = max(1, len(records))
    label_rows = []
    excluded = excluded_label_rules(args)
    for rule_name in RULE_NAMES:
        valid_count = valid_counts[rule_name]
        label_ratio = label_counts[rule_name] / total_labels
        eligible_label_count = eligible_counts[rule_name]
        label_rows.append(
            {
                "rule_name": rule_name,
                "label_count": label_counts[rule_name],
                "label_ratio": label_ratio,
                "valid_count": valid_count,
                "valid_ratio": valid_count / total_states,
                "mean_proxy_score": proxy_sums[rule_name] / max(1, valid_count),
                "is_excluded_from_label": rule_name in excluded,
                "eligible_label_count": eligible_label_count,
                "eligible_label_ratio": eligible_label_count / total_states,
            }
        )
    write_csv(label_rows, paths["label_stats"], LABEL_FIELDS)
    ranker_ratio = label_counts["mlp_ranker_soft_ce"] / total_labels
    warnings = []
    if ranker_ratio < 0.35:
        warnings.append(f"WARNING: mlp_ranker_soft_ce label_ratio={ranker_ratio:.4f} < 0.35")
    for rule_name in RULE_NAMES:
        if rule_name != args.baseline_rule and label_counts[rule_name] / total_labels > 0.5:
            warnings.append(f"WARNING: non-baseline rule {rule_name} label_ratio={label_counts[rule_name] / total_labels:.4f} > 0.5")
    active_label_rules = [rule for rule in RULE_NAMES if label_counts[rule] > 0]
    if len(active_label_rules) <= 1:
        warnings.append(f"WARNING: labels concentrate in {len(active_label_rules)} rule(s): {active_label_rules}")
    for message in warnings:
        print(message)
    paths["manifest"].write_text(
        json.dumps(
            {
                "run_id": run_id,
                "start_time": datetime.now().isoformat(timespec="seconds"),
                "command_args": vars(args),
                "dataset_path": str(paths["dataset"]),
                "exclude_label_rules": sorted(excluded),
                "warnings": warnings,
                "python_version": sys.version,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved Rule Library BC dataset to {paths['dataset']}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, default="small")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--ranker_ckpt", default="checkpoints/stage_C/mlp_ranker/small_topk5_soft_ce/best.pt")
    parser.add_argument(
        "--label_mode",
        choices=["one_step_cmax", "one_step_ect", "conservative_margin"],
        default="one_step_cmax",
    )
    parser.add_argument("--baseline_rule", choices=RULE_NAMES, default="mlp_ranker_soft_ce")
    parser.add_argument("--label_margin", type=float, default=0.5)
    parser.add_argument("--exclude_label_rules", nargs="*", choices=RULE_NAMES, default=[])
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--output_dir", default="data/processed/stage_F/rule_library_bc")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()

