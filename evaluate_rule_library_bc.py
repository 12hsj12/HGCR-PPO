"""Evaluate a Rule Library BC checkpoint on a rule-level dataset."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from instance_manager import SIZES
from src.rules.experience_rule_library import RULE_NAMES
from train_rule_library_bc import RuleBCDataset, RuleBCSelector, collate
from utils.experiment_io import write_csv


SUMMARY_FIELDS = [
    "run_id",
    "size",
    "top_k",
    "dataset_path",
    "ckpt_path",
    "num_samples",
    "accuracy",
    "macro_f1",
    "label_distribution",
    "prediction_distribution",
    "per_rule_precision",
    "per_rule_recall",
    "per_rule_f1",
]
DIST_FIELDS = [
    "rule_id",
    "rule_name",
    "label_count",
    "label_ratio",
    "prediction_count",
    "prediction_ratio",
    "precision",
    "recall",
    "f1",
]


def make_run_id(args) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"RLBCEval_{args.size}_k{args.top_k}_{stamp}_{uuid.uuid4().hex[:8]}"


def output_paths(args, run_id: str) -> Dict[str, Path]:
    run_dir = Path(args.output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return {
        "run_dir": run_dir,
        "summary": run_dir / "eval_summary.csv",
        "distribution": run_dir / "label_pred_distribution.csv",
        "manifest": run_dir / "manifest.json",
    }


def load_model(ckpt_path: str, fallback_input_dim: int, device) -> RuleBCSelector:
    checkpoint = torch.load(ckpt_path, map_location=device)
    input_dim = int(checkpoint.get("input_dim", fallback_input_dim))
    action_dim = int(checkpoint.get("action_dim", len(RULE_NAMES)))
    model = RuleBCSelector(input_dim, action_dim=action_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def metrics(labels: List[int], preds: List[int]) -> Dict:
    total = max(1, len(labels))
    label_counts = Counter(labels)
    pred_counts = Counter(preds)
    per_rule = {}
    f1_values = []
    rows = []
    for idx, rule_name in enumerate(RULE_NAMES):
        tp = sum(1 for y, p in zip(labels, preds) if y == idx and p == idx)
        fp = sum(1 for y, p in zip(labels, preds) if y != idx and p == idx)
        fn = sum(1 for y, p in zip(labels, preds) if y == idx and p != idx)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        f1_values.append(f1)
        per_rule[rule_name] = {"precision": precision, "recall": recall, "f1": f1}
        rows.append(
            {
                "rule_id": idx,
                "rule_name": rule_name,
                "label_count": label_counts[idx],
                "label_ratio": label_counts[idx] / total,
                "prediction_count": pred_counts[idx],
                "prediction_ratio": pred_counts[idx] / total,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "accuracy": sum(1 for y, p in zip(labels, preds) if y == p) / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "label_distribution": {str(k): v for k, v in sorted(label_counts.items())},
        "prediction_distribution": {str(k): v for k, v in sorted(pred_counts.items())},
        "per_rule": per_rule,
        "distribution_rows": rows,
    }


def warnings_from_distribution(distribution_rows: List[Dict]) -> List[str]:
    warnings = []
    by_rule = {row["rule_name"]: row for row in distribution_rows}
    ranker_ratio = by_rule["mlp_ranker_soft_ce"]["prediction_ratio"]
    lookahead_ratio = by_rule["lookahead"]["prediction_ratio"]
    if ranker_ratio < 0.35:
        warnings.append(f"WARNING: mlp_ranker_soft_ce prediction_ratio={ranker_ratio:.4f} < 0.35")
    if lookahead_ratio > 0.55:
        warnings.append(f"WARNING: lookahead prediction_ratio={lookahead_ratio:.4f} > 0.55")
    for row in distribution_rows:
        if row["rule_name"] != "mlp_ranker_soft_ce" and row["prediction_ratio"] > 0.55:
            warnings.append(
                f"WARNING: non-baseline rule {row['rule_name']} prediction_ratio={row['prediction_ratio']:.4f} > 0.55"
            )
    return warnings


def evaluate(args) -> Dict[str, Path]:
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = torch.load(args.dataset_path, map_location="cpu")
    records = dataset["records"]
    if args.smoke_test:
        records = records[: min(len(records), 32)]
    if not records:
        raise RuntimeError("Dataset contains no records.")
    loader = DataLoader(RuleBCDataset(records), batch_size=256, shuffle=False, collate_fn=collate)
    model = load_model(args.ckpt_path, len(records[0]["state_features"]), device)

    labels: List[int] = []
    preds: List[int] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            masks = batch["masks"].to(device)
            logits = model(features).masked_fill(~masks, -1e9)
            labels.extend(batch["labels"].tolist())
            preds.extend(logits.argmax(dim=1).cpu().tolist())

    result = metrics(labels, preds)
    warnings = warnings_from_distribution(result["distribution_rows"])
    for message in warnings:
        print(message)

    run_id = make_run_id(args)
    paths = output_paths(args, run_id)
    summary_row = {
        "run_id": run_id,
        "size": args.size,
        "top_k": args.top_k,
        "dataset_path": args.dataset_path,
        "ckpt_path": args.ckpt_path,
        "num_samples": len(records),
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "label_distribution": json.dumps(result["label_distribution"], sort_keys=True),
        "prediction_distribution": json.dumps(result["prediction_distribution"], sort_keys=True),
        "per_rule_precision": json.dumps({k: v["precision"] for k, v in result["per_rule"].items()}, sort_keys=True),
        "per_rule_recall": json.dumps({k: v["recall"] for k, v in result["per_rule"].items()}, sort_keys=True),
        "per_rule_f1": json.dumps({k: v["f1"] for k, v in result["per_rule"].items()}, sort_keys=True),
    }
    write_csv([summary_row], paths["summary"], SUMMARY_FIELDS)
    write_csv(result["distribution_rows"], paths["distribution"], DIST_FIELDS)
    paths["manifest"].write_text(
        json.dumps(
            {
                "run_id": run_id,
                "args": vars(args),
                "warnings": warnings,
                "python_version": sys.version,
                "output_files": {key: str(value) for key, value in paths.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved Rule Library BC eval outputs to {paths['run_dir']}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--size", choices=SIZES, default="small")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", default="data/results/stage_F/rule_library_bc_eval")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

