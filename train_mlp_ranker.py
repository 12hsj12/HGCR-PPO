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
from mlp_models import CandidateScorer, save_checkpoint
from stage_c_utils import dataset_path, load_ranker_records


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
    for row_idx, row in enumerate(batch):
        count = len(row["candidate_features"])
        features[row_idx, :count] = torch.tensor(row["candidate_features"], dtype=torch.float32)
        mask[row_idx, :count] = True
        cmax[row_idx, :count] = torch.tensor(row["oracle_cmax_per_candidate"], dtype=torch.float32)
        labels[row_idx] = int(row["best_candidate_index"])
    return {"features": features, "mask": mask, "cmax": cmax, "labels": labels}


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
            },
            {
                "candidate_features": [[0.2, 0.4, 0.1], [0.1, 0.5, 0.2]],
                "oracle_cmax_per_candidate": [14.0, 11.0],
                "best_candidate_index": 1,
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
    dry_run: bool = False,
    data_dir: str = "data/ranker_dataset/",
) -> None:
    train_records = _load_records(size, top_k, "train", data_dir, dry_run)
    val_records = _load_records(size, top_k, "val", data_dir, dry_run)
    if dry_run and len(val_records[0]["candidate_features"][0]) != len(train_records[0]["candidate_features"][0]):
        val_records = train_records[:]
    input_dim = len(train_records[0]["candidate_features"][0])
    model = CandidateScorer(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(RankerRecordDataset(train_records), batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(RankerRecordDataset(val_records), batch_size=batch_size, shuffle=False, collate_fn=_collate)

    experiment_name = f"{size}_topk{top_k}_{loss_type}"
    log_dir = Path("logs/stage_C/mlp_ranker")
    ckpt_dir = Path("checkpoints/stage_C/mlp_ranker") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{experiment_name}.csv"
    checkpoint_metadata = {"size": size, "top_k": top_k, "loss_type": loss_type, "type": "mlp_ranker"}

    best_val = float("inf")
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_top1_acc"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for batch in train_loader:
                scores = model(batch["features"])
                loss = _loss(scores, batch, loss_type, temperature)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach()))
            val_loss, val_acc = evaluate(model, val_loader, loss_type, temperature)
            train_loss = sum(train_losses) / len(train_losses)
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_top1_acc": val_acc})
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(ckpt_dir / "best.pt", model, input_dim, checkpoint_metadata)
            if dry_run:
                break

    save_checkpoint(ckpt_dir / "last.pt", model, input_dim, checkpoint_metadata)
    print(f"Saved MLP-Ranker checkpoints to {ckpt_dir}")
    print(f"Saved training log to {log_path}")


def _loss(scores: torch.Tensor, batch: Dict[str, torch.Tensor], loss_type: str, temperature: float) -> torch.Tensor:
    if loss_type == "pairwise":
        return pairwise_ranking_loss(scores, batch["cmax"], batch["mask"])
    return soft_label_cross_entropy(scores, batch["cmax"], batch["mask"], temperature)


def evaluate(model: CandidateScorer, loader: DataLoader, loss_type: str, temperature: float) -> tuple[float, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            scores = model(batch["features"]).masked_fill(~batch["mask"], -1e9)
            loss = _loss(scores, batch, loss_type, temperature)
            losses.append(float(loss))
            correct += int((scores.argmax(dim=1) == batch["labels"]).sum())
            total += len(batch["labels"])
    return sum(losses) / len(losses), correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--loss_type", choices=["soft_ce", "pairwise"], default="soft_ce")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--data_dir", default="data/ranker_dataset/")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
