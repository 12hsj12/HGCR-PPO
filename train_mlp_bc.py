"""Train an MLP behavior-cloning classifier over HybridTopK candidates."""

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
    labels = torch.zeros(len(batch), dtype=torch.long)
    for row_idx, row in enumerate(batch):
        count = len(row["candidate_features"])
        features[row_idx, :count] = torch.tensor(row["candidate_features"], dtype=torch.float32)
        mask[row_idx, :count] = True
        labels[row_idx] = int(row["best_candidate_index"])
    return {"features": features, "mask": mask, "labels": labels}


def _load_records(size: str, top_k: int, split: str, data_dir: str, dry_run: bool) -> List[Dict]:
    path = dataset_path(data_dir, size, split, top_k)
    if path.exists():
        records = load_ranker_records(path)
    elif dry_run:
        records = [
            {
                "candidate_features": [[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]],
                "best_candidate_index": 0,
            },
            {
                "candidate_features": [[0.2, 0.4, 0.1], [0.1, 0.5, 0.2]],
                "best_candidate_index": 1,
            },
        ]
    else:
        raise FileNotFoundError(f"Missing dataset {path}. Run generate_ranker_dataset.py first.")
    return records[: min(len(records), 8)] if dry_run else records


def train(
    size: str,
    top_k: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_train_samples: int | None = None,
    dry_run: bool = False,
    data_dir: str = "data/ranker_dataset/",
    run_id: str | None = None,
    overwrite: bool = False,
) -> None:
    train_records = _load_records(size, top_k, "train", data_dir, dry_run)
    val_records = _load_records(size, top_k, "val", data_dir, dry_run)
    if max_train_samples is not None:
        train_records = train_records[: max(0, max_train_samples)]
    if dry_run and len(val_records[0]["candidate_features"][0]) != len(train_records[0]["candidate_features"][0]):
        val_records = train_records[:]

    input_dim = len(train_records[0]["candidate_features"][0])
    model = CandidateScorer(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(RankerRecordDataset(train_records), batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(RankerRecordDataset(val_records), batch_size=batch_size, shuffle=False, collate_fn=_collate)

    resolved_run_id = make_run_id(run_id)
    log_dir = Path("logs/stage_C/mlp_bc")
    ckpt_dir = make_run_dir(
        Path("checkpoints/stage_C/mlp_bc"),
        [size, f"topk{top_k}"],
        f"runid{resolved_run_id}",
        overwrite=overwrite,
    )
    latest_ckpt_dir = Path("checkpoints/stage_C/mlp_bc") / f"{size}_topk{top_k}_latest"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = make_result_path(
        log_dir,
        "train",
        [size, f"topk{top_k}", f"runid{resolved_run_id}"],
        run_id=None,
        overwrite=overwrite,
    )
    checkpoint_metadata = {"size": size, "top_k": top_k, "run_id": resolved_run_id, "type": "mlp_bc"}

    best_val = float("inf")
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc"])
        writer.writeheader()
        desc = f"train-mlp-bc {size} topk{top_k}"
        for epoch in progress_iter(range(1, epochs + 1), desc=desc, total=epochs):
            model.train()
            train_losses = []
            for batch in train_loader:
                scores = model(batch["features"])
                scores = scores.masked_fill(~batch["mask"], -1e9)
                loss = F.cross_entropy(scores, batch["labels"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach()))

            val_loss, val_acc = evaluate(model, val_loader)
            train_loss = sum(train_losses) / len(train_losses)
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(ckpt_dir / "best.pt", model, input_dim, checkpoint_metadata)
            if dry_run:
                break

    save_checkpoint(ckpt_dir / "last.pt", model, input_dim, checkpoint_metadata)
    update_latest_dir(ckpt_dir, latest_ckpt_dir)
    print(f"Saved MLP-BC checkpoints to {ckpt_dir}")
    print(f"Updated MLP-BC latest checkpoints to {latest_ckpt_dir}")
    print(f"Saved training log to {log_path}")


def evaluate(model: CandidateScorer, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            scores = model(batch["features"])
            scores = scores.masked_fill(~batch["mask"], -1e9)
            loss = F.cross_entropy(scores, batch["labels"])
            losses.append(float(loss))
            correct += int((scores.argmax(dim=1) == batch["labels"]).sum())
            total += len(batch["labels"])
    return sum(losses) / len(losses), correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--data_dir", default="data/ranker_dataset/")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
