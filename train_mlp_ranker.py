"""Train an MLP ranker over HybridTopK candidate sets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from instance_manager import SIZES
from mlp_models import CandidateScorer, ImprovementAwareCandidateScorer, save_checkpoint
from stage_c_utils import dataset_path, load_ranker_records
from utils.experiment_io import make_result_path, make_run_dir, make_run_id, update_latest_dir, progress_iter


class RankerRecordDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return self.records[idx]


def _collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_candidates = max(len(row["candidate_features"]) for row in batch)
    feature_dim = len(batch[0]["candidate_features"][0])
    features = torch.zeros(len(batch), max_candidates, feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_candidates, dtype=torch.bool)
    cmax = torch.zeros(len(batch), max_candidates, dtype=torch.float32)
    labels = torch.zeros(len(batch), dtype=torch.long)
    gate_labels = torch.zeros(len(batch), dtype=torch.float32)
    for row_idx, row in enumerate(batch):
        count = len(row["candidate_features"])
        features[row_idx, :count] = torch.tensor(row["candidate_features"], dtype=torch.float32)
        mask[row_idx, :count] = True
        cmax[row_idx, :count] = torch.tensor(row["oracle_cmax_per_candidate"], dtype=torch.float32)
        labels[row_idx] = int(row["best_candidate_index"])
        gate_labels[row_idx] = float(row.get("is_improvement_state", 0))
    return {"features": features, "mask": mask, "cmax": cmax, "labels": labels, "gate_labels": gate_labels}


def _load_records(size: str, top_k: int, split: str, data_dir: str, dry_run: bool) -> List[Dict]:
    path = dataset_path(data_dir, size, split, top_k)
    if path.exists():
        records = load_ranker_records(path)
    elif dry_run:
        records = [
            {
                "candidate_features": [[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]],
                "oracle_cmax_per_candidate": [10.0, 12.0],
                "best_candidate_index": 0,
                "is_improvement_state": 0,
            },
            {
                "candidate_features": [[0.2, 0.4, 0.1], [0.1, 0.5, 0.2]],
                "oracle_cmax_per_candidate": [14.0, 11.0],
                "best_candidate_index": 1,
                "is_improvement_state": 1,
            },
        ]
    else:
        raise FileNotFoundError(f"Missing dataset {path}. Run generate_ranker_dataset.py first.")
    return records[: min(len(records), 8)] if dry_run else records


def soft_label_cross_entropy(scores: torch.Tensor, cmax: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    masked_scores = scores.masked_fill(~mask, -1e9)
    masked_cmax = cmax.masked_fill(~mask, 1e9)
    labels = torch.softmax(-masked_cmax / max(temperature, 1e-6), dim=1)
    return -(labels * F.log_softmax(masked_scores, dim=1)).sum(dim=1).mean()


def pairwise_ranking_loss(scores: torch.Tensor, cmax: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for row_scores, row_cmax, row_mask in zip(scores, cmax, mask):
        valid_scores = row_scores[row_mask]
        valid_cmax = row_cmax[row_mask]
        for i in range(len(valid_scores)):
            for j in range(len(valid_scores)):
                if valid_cmax[i] + 1e-9 < valid_cmax[j]:
                    losses.append(F.softplus(-(valid_scores[i] - valid_scores[j])))
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def train(
    size: str,
    top_k: int,
    loss_type: str,
    temperature: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lambda_gate: float = 1.0,
    improvement_epsilon: float = 0.0,
    dry_run: bool = False,
    data_dir: str = "data/ranker_dataset/",
    run_id: str | None = None,
    overwrite: bool = False,
) -> None:
    train_records = _load_records(size, top_k, "train", data_dir, dry_run)
    val_records = _load_records(size, top_k, "val", data_dir, dry_run)
    if dry_run and len(val_records[0]["candidate_features"][0]) != len(train_records[0]["candidate_features"][0]):
        val_records = train_records[:]
    input_dim = len(train_records[0]["candidate_features"][0])
    model = ImprovementAwareCandidateScorer(input_dim) if loss_type == "improvement_aware" else CandidateScorer(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(RankerRecordDataset(train_records), batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(RankerRecordDataset(val_records), batch_size=batch_size, shuffle=False, collate_fn=_collate)

    resolved_run_id = make_run_id(run_id)
    if loss_type == "improvement_aware":
        log_dir = Path("logs/stage_E/improvement_ranker")
        ckpt_dir = make_run_dir(
            Path("checkpoints/stage_E/improvement_ranker"),
            [size, f"topk{top_k}", f"eps{improvement_epsilon}"],
            f"runid{resolved_run_id}",
            overwrite=overwrite,
        )
        latest_ckpt_dir = Path("checkpoints/stage_E/improvement_ranker") / (
            f"{size}_topk{top_k}_eps{improvement_epsilon}_latest"
        )
        log_tokens = [size, f"topk{top_k}", f"eps{improvement_epsilon}", f"runid{resolved_run_id}"]
    else:
        log_dir = Path("logs/stage_C/mlp_ranker")
        ckpt_dir = make_run_dir(
            Path("checkpoints/stage_C/mlp_ranker"),
            [size, f"topk{top_k}", loss_type],
            f"runid{resolved_run_id}",
            overwrite=overwrite,
        )
        latest_ckpt_dir = Path("checkpoints/stage_C/mlp_ranker") / f"{size}_topk{top_k}_{loss_type}_latest"
        log_tokens = [size, f"topk{top_k}", loss_type, f"runid{resolved_run_id}"]
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = make_result_path(
        log_dir,
        "train",
        log_tokens,
        run_id=None,
        overwrite=overwrite,
    )
    checkpoint_metadata = {
        "size": size,
        "top_k": top_k,
        "loss_type": loss_type,
        "run_id": resolved_run_id,
        "type": "improvement_aware_ranker" if loss_type == "improvement_aware" else "mlp_ranker",
        "improvement_epsilon": improvement_epsilon,
        "lambda_gate": lambda_gate,
    }

    best_val = float("inf")
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_top1_acc", "val_gate_acc"])
        writer.writeheader()
        desc = f"train-mlp-ranker {size} topk{top_k} {loss_type}"
        for epoch in progress_iter(range(1, epochs + 1), desc=desc, total=epochs):
            model.train()
            train_losses = []
            for batch in train_loader:
                scores, gate_logits = _model_outputs(model, batch)
                loss = _loss(scores, batch, loss_type, temperature, gate_logits=gate_logits, lambda_gate=lambda_gate)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach()))
            val_loss, val_acc, val_gate_acc = evaluate(model, val_loader, loss_type, temperature, lambda_gate)
            train_loss = sum(train_losses) / len(train_losses)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_top1_acc": val_acc,
                    "val_gate_acc": val_gate_acc,
                }
            )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(ckpt_dir / "best.pt", model, input_dim, checkpoint_metadata)
            if dry_run:
                break

    save_checkpoint(ckpt_dir / "last.pt", model, input_dim, checkpoint_metadata)
    update_latest_dir(ckpt_dir, latest_ckpt_dir)
    print(f"Saved MLP-Ranker checkpoints to {ckpt_dir}")
    print(f"Updated MLP-Ranker latest checkpoints to {latest_ckpt_dir}")
    print(f"Saved training log to {log_path}")


def _model_outputs(model: CandidateScorer, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if hasattr(model, "forward_with_gate"):
        return model.forward_with_gate(batch["features"], batch["mask"])
    return model(batch["features"]), None


def _loss(
    scores: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    loss_type: str,
    temperature: float,
    gate_logits: torch.Tensor | None = None,
    lambda_gate: float = 1.0,
) -> torch.Tensor:
    if loss_type == "improvement_aware":
        if gate_logits is None:
            raise ValueError("improvement_aware loss requires a model with improvement_gate_head.")
        rank_loss = soft_label_cross_entropy(scores, batch["cmax"], batch["mask"], temperature)
        gate_loss = F.binary_cross_entropy_with_logits(gate_logits, batch["gate_labels"])
        return rank_loss + lambda_gate * gate_loss
    if loss_type == "pairwise":
        return pairwise_ranking_loss(scores, batch["cmax"], batch["mask"])
    return soft_label_cross_entropy(scores, batch["cmax"], batch["mask"], temperature)


def evaluate(
    model: CandidateScorer,
    loader: DataLoader,
    loss_type: str,
    temperature: float,
    lambda_gate: float = 1.0,
) -> tuple[float, float, float]:
    model.eval()
    losses = []
    correct = 0
    gate_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            raw_scores, gate_logits = _model_outputs(model, batch)
            scores = raw_scores.masked_fill(~batch["mask"], -1e9)
            loss = _loss(scores, batch, loss_type, temperature, gate_logits=gate_logits, lambda_gate=lambda_gate)
            losses.append(float(loss))
            correct += int((scores.argmax(dim=1) == batch["labels"]).sum())
            if gate_logits is not None:
                gate_pred = (torch.sigmoid(gate_logits) >= 0.5).to(batch["gate_labels"].dtype)
                gate_correct += int((gate_pred == batch["gate_labels"]).sum())
            total += len(batch["labels"])
    return sum(losses) / len(losses), correct / max(1, total), gate_correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--loss_type", choices=["soft_ce", "pairwise", "improvement_aware"], default="soft_ce")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambda_gate", type=float, default=1.0)
    parser.add_argument("--improvement_epsilon", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--data_dir", default="data/ranker_dataset/")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
