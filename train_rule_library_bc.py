"""Train a rule-library behavior cloning selector."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from copy import deepcopy
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from instance_manager import SIZES
from src.rules.experience_rule_library import RULE_NAMES
from utils.experiment_io import write_csv


TRAIN_FIELDS = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"]
SUMMARY_FIELDS = [
    "run_id",
    "size",
    "top_k",
    "best_val_loss",
    "best_val_accuracy",
    "best_val_macro_f1",
    "label_distribution",
    "prediction_distribution",
    "per_rule_metrics",
]


class RuleBCDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return self.records[idx]


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    features = torch.tensor([row["state_features"] for row in batch], dtype=torch.float32)
    labels = torch.tensor([int(row["label_rule_id"]) for row in batch], dtype=torch.long)
    masks = torch.tensor([row["rule_valid_mask"] for row in batch], dtype=torch.bool)
    return {"features": features, "labels": labels, "masks": masks}


class RuleBCSelector(nn.Module):
    def __init__(self, input_dim: int, action_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def make_run_id(args) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"RuleLibBC_{args.size}_k{args.top_k}_ep{args.epochs}_{stamp}_{uuid.uuid4().hex[:8]}"


def output_paths(args, run_id: str) -> Dict[str, Path]:
    ckpt_dir = Path(args.output_dir) / run_id
    result_dir = Path("data/results/stage_F/rule_library_bc/runs") / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    return {
        "best": ckpt_dir / "best.pt",
        "last": ckpt_dir / "last.pt",
        "train_log": result_dir / f"train_log__{run_id}.csv",
        "eval_summary": result_dir / f"eval_summary__{run_id}.csv",
        "manifest": result_dir / f"manifest__{run_id}.json",
    }


def evaluate(model, loader, device) -> Dict:
    model.eval()
    losses = []
    labels_all = []
    preds_all = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            masks = batch["masks"].to(device)
            logits = model(features).masked_fill(~masks, -1e9)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)
            losses.append(float(loss.item()))
            labels_all.extend(labels.cpu().tolist())
            preds_all.extend(preds.cpu().tolist())
    metrics = classification_metrics(labels_all, preds_all)
    metrics["loss"] = sum(losses) / max(1, len(losses))
    return metrics


def classification_metrics(labels: List[int], preds: List[int]) -> Dict:
    total = max(1, len(labels))
    accuracy = sum(1 for y, p in zip(labels, preds) if y == p) / total
    per_rule = {}
    f1s = []
    for idx, rule in enumerate(RULE_NAMES):
        tp = sum(1 for y, p in zip(labels, preds) if y == idx and p == idx)
        fp = sum(1 for y, p in zip(labels, preds) if y != idx and p == idx)
        fn = sum(1 for y, p in zip(labels, preds) if y == idx and p != idx)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        f1s.append(f1)
        per_rule[rule] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / len(f1s),
        "per_rule_metrics": per_rule,
        "label_distribution": dict(Counter(labels)),
        "prediction_distribution": dict(Counter(preds)),
    }


def save_checkpoint(path: Path, model, input_dim: int, run_id: str, args) -> None:
    if path.exists():
        raise FileExistsError(path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "action_dim": len(RULE_NAMES),
            "rule_names": RULE_NAMES,
            "metadata": {"run_id": run_id, "size": args.size, "top_k": args.top_k, "type": "rule_library_bc"},
        },
        path,
    )


def train(args) -> None:
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = torch.load(args.dataset_path, map_location="cpu")
    records = dataset["records"]
    if args.smoke_test:
        records = records[: min(len(records), 16)]
    split = max(1, int(0.8 * len(records)))
    train_records = records[:split]
    val_records = records[split:] or records[:]
    train_loader = DataLoader(RuleBCDataset(train_records), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(RuleBCDataset(val_records), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    input_dim = len(records[0]["state_features"])
    model = RuleBCSelector(input_dim, action_dim=len(RULE_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    run_id = make_run_id(args)
    paths = output_paths(args, run_id)
    best = {"loss": float("inf"), "accuracy": 0.0, "macro_f1": 0.0}
    best_state = None
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            masks = batch["masks"].to(device)
            logits = model(features).masked_fill(~masks, -1e9)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val = evaluate(model, val_loader, device)
        if val["loss"] < best["loss"]:
            best = val
            best_state = deepcopy(model.state_dict())
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": sum(losses) / max(1, len(losses)),
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
                "val_macro_f1": val["macro_f1"],
            }
        )
        if args.smoke_test:
            break
    last_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(paths["best"], model, input_dim, run_id, args)
    model.load_state_dict(last_state)
    save_checkpoint(paths["last"], model, input_dim, run_id, args)
    write_csv(log_rows, paths["train_log"], TRAIN_FIELDS)
    write_csv(
        [
            {
                "run_id": run_id,
                "size": args.size,
                "top_k": args.top_k,
                "best_val_loss": best["loss"],
                "best_val_accuracy": best["accuracy"],
                "best_val_macro_f1": best["macro_f1"],
                "label_distribution": json.dumps(best["label_distribution"], sort_keys=True),
                "prediction_distribution": json.dumps(best["prediction_distribution"], sort_keys=True),
                "per_rule_metrics": json.dumps(best["per_rule_metrics"], sort_keys=True),
            }
        ],
        paths["eval_summary"],
        SUMMARY_FIELDS,
    )
    paths["manifest"].write_text(
        json.dumps({"run_id": run_id, "args": vars(args), "python_version": sys.version}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved Rule Library BC checkpoint to {paths['best']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--size", choices=SIZES, default="small")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--output_dir", default="checkpoints/stage_F/rule_library_bc")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
