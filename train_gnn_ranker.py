"""Train the Stage D GNN-Ranker on HybridTopK oracle ranking data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gnn_graph_builder import build_graph_from_env
from gnn_ranker_models import BipartiteGNNRanker, graph_to_torch, save_checkpoint
from instance_manager import SIZES, load_fixed_instances
from src.baselines.heuristics import choose_split_num
from src.envs.rolling_scheduling_env import RollingSchedulingEnv
from stage_c_utils import dataset_path, hybrid_candidates, load_ranker_records, oracle_cmax_per_candidate, best_candidate_index
from utils.experiment_io import make_result_path, make_run_dir, make_run_id, progress_iter, update_latest_dir


class GNNRankerDataset(Dataset):
    def __init__(self, records: List[Dict], instances_by_id: Dict[str, object], top_k: int, device: str):
        self.records = records
        self.instances_by_id = instances_by_id
        self.top_k = top_k
        self.device = device

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        env = reconstruct_env_for_record(record, self.instances_by_id)
        candidates = list(record.get("candidate_job_ids") or hybrid_candidates(env, self.top_k))
        graph = build_graph_from_env(env, candidates)
        return {
            "graph": graph_to_torch(graph, self.device),
            "cmax": torch.tensor(record["oracle_cmax_per_candidate"], dtype=torch.float32, device=self.device),
            "label": int(record["best_candidate_index"]),
        }


def reconstruct_env_for_record(record: Dict, instances_by_id: Dict[str, object]):
    instance_id = record.get("instance_id")
    if instance_id not in instances_by_id:
        raise KeyError(f"Instance {instance_id!r} is not available for GNN graph reconstruction.")
    env = RollingSchedulingEnv(instances_by_id[instance_id])
    env.reset(instances_by_id[instance_id])
    history = record.get("rollout_job_history") or record.get("scheduled_job_history") or []
    for job_id in history:
        env.step((job_id, choose_split_num(env, job_id)))
    return env


def _collate(batch: List[Dict]) -> Dict:
    max_candidates = max(len(row["cmax"]) for row in batch)
    device = batch[0]["cmax"].device
    cmax = torch.zeros((len(batch), max_candidates), dtype=torch.float32, device=device)
    mask = torch.zeros((len(batch), max_candidates), dtype=torch.bool, device=device)
    labels = torch.zeros(len(batch), dtype=torch.long, device=device)
    for row_idx, row in enumerate(batch):
        count = len(row["cmax"])
        cmax[row_idx, :count] = row["cmax"]
        mask[row_idx, :count] = True
        labels[row_idx] = row["label"]
    return {"graphs": [row["graph"] for row in batch], "cmax": cmax, "mask": mask, "labels": labels}


def soft_label_cross_entropy(scores: torch.Tensor, cmax: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    masked_scores = scores.masked_fill(~mask, -1e9)
    masked_cmax = cmax.masked_fill(~mask, 1e9)
    labels = torch.softmax(-masked_cmax / max(temperature, 1e-6), dim=1)
    return -(labels * F.log_softmax(masked_scores, dim=1)).sum(dim=1).mean()


def _load_records(size: str, top_k: int, split: str, data_dir: str, dry_run: bool, max_samples: int | None) -> List[Dict]:
    path = dataset_path(data_dir, size, split, top_k)
    if path.exists():
        records = load_ranker_records(path)
        records = _attach_inferred_histories(records)
    elif dry_run:
        records = _dry_run_records(size, top_k)
    else:
        raise FileNotFoundError(f"Missing dataset {path}. Run generate_ranker_dataset.py first.")
    if dry_run:
        records = records[: min(len(records), max_samples or 8, 8)]
    elif max_samples is not None:
        records = records[: max(0, max_samples)]
    return records


def _attach_inferred_histories(records: List[Dict]) -> List[Dict]:
    histories: Dict[str, List[str]] = {}
    enriched = []
    for record in sorted(records, key=lambda row: (str(row.get("instance_id")), int(row.get("step_id", 0)))):
        item = dict(record)
        instance_id = str(item.get("instance_id"))
        if "rollout_job_history" not in item and "scheduled_job_history" not in item:
            item["rollout_job_history"] = list(histories.get(instance_id, []))
        histories.setdefault(instance_id, [])
        if item.get("best_job_id"):
            histories[instance_id].append(item["best_job_id"])
        enriched.append(item)
    return enriched


def _dry_run_records(size: str, top_k: int) -> List[Dict]:
    instances = load_fixed_instances(size, "train")[:1]
    env = RollingSchedulingEnv(instances[0])
    env.reset(instances[0])
    records: List[Dict] = []
    history: List[str] = []
    for step_id in range(2):
        candidates = hybrid_candidates(env, top_k)
        cmax_values = oracle_cmax_per_candidate(env, candidates)
        best_idx = best_candidate_index(cmax_values)
        records.append(
            {
                "instance_id": getattr(instances[0], "instance_id", instances[0].name),
                "step_id": step_id,
                "rollout_job_history": list(history),
                "candidate_job_ids": list(candidates),
                "oracle_cmax_per_candidate": [float(v) for v in cmax_values],
                "best_candidate_index": best_idx,
                "best_job_id": candidates[best_idx],
            }
        )
        env.step((candidates[best_idx], choose_split_num(env, candidates[best_idx])))
        history.append(candidates[best_idx])
    return records


def _load_instances(size: str) -> Dict[str, object]:
    instances = load_fixed_instances(size, "train") + load_fixed_instances(size, "val")
    return {getattr(instance, "instance_id", instance.name): instance for instance in instances}


def _infer_dims(dataset: GNNRankerDataset) -> tuple[int, int, int]:
    sample = dataset[0]["graph"]
    return (
        int(sample["job_features"].shape[1]),
        int(sample["machine_features"].shape[1]),
        int(sample["edge_features"].shape[1]),
    )


def train(
    size: str,
    top_k: int,
    loss_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    run_id: str | None = None,
    dry_run: bool = False,
    max_train_samples: int | None = None,
    data_dir: str = "data/ranker_dataset/",
    temperature: float = 1.0,
    hidden_dim: int = 128,
    num_layers: int = 2,
    device: str = "cpu",
    overwrite: bool = False,
) -> None:
    if loss_type != "soft_ce":
        raise ValueError("Stage D first version only supports --loss_type soft_ce.")
    train_records = _load_records(size, top_k, "train", data_dir, dry_run, max_train_samples)
    val_records = _load_records(size, top_k, "val", data_dir, dry_run, max_train_samples if dry_run else None)
    instances_by_id = _load_instances(size)
    train_dataset = GNNRankerDataset(train_records, instances_by_id, top_k, device)
    val_dataset = GNNRankerDataset(val_records, instances_by_id, top_k, device)
    job_dim, machine_dim, edge_dim = _infer_dims(train_dataset)
    model = BipartiteGNNRanker(job_dim, machine_dim, edge_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    resolved_run_id = make_run_id(run_id)
    ckpt_dir = make_run_dir(
        Path("checkpoints/stage_D/gnn_ranker"),
        [size, f"topk{top_k}", loss_type],
        f"runid{resolved_run_id}",
        overwrite=overwrite,
    )
    latest_ckpt_dir = Path("checkpoints/stage_D/gnn_ranker") / f"{size}_topk{top_k}_{loss_type}_latest"
    log_dir = Path("logs/stage_D/gnn_ranker")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = make_result_path(
        log_dir,
        "train",
        [size, f"topk{top_k}", loss_type, f"runid{resolved_run_id}"],
        run_id=None,
        overwrite=overwrite,
    )
    metadata = {
        "size": size,
        "top_k": top_k,
        "loss_type": loss_type,
        "run_id": resolved_run_id,
        "type": "gnn_ranker",
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }

    best_val = float("inf")
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_top1_acc"])
        writer.writeheader()
        desc = f"train-gnn-ranker {size} topk{top_k} {loss_type}"
        for epoch in progress_iter(range(1, epochs + 1), desc=desc, total=epochs):
            model.train()
            losses = []
            for batch in train_loader:
                scores, score_mask = model(batch["graphs"])
                mask = batch["mask"][:, : scores.shape[1]] & score_mask
                loss = soft_label_cross_entropy(scores, batch["cmax"][:, : scores.shape[1]], mask, temperature)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            val_loss, val_acc = evaluate(model, val_loader, temperature)
            train_loss = sum(losses) / max(1, len(losses))
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_top1_acc": val_acc})
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(ckpt_dir / "best.pt", model, metadata)
            if dry_run:
                break

    save_checkpoint(ckpt_dir / "last.pt", model, metadata)
    update_latest_dir(ckpt_dir, latest_ckpt_dir)
    print(f"Saved GNN-Ranker checkpoints to {ckpt_dir}")
    print(f"Updated GNN-Ranker latest checkpoints to {latest_ckpt_dir}")
    print(f"Saved training log to {log_path}")


def evaluate(model: BipartiteGNNRanker, loader: DataLoader, temperature: float) -> tuple[float, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            scores, score_mask = model(batch["graphs"])
            mask = batch["mask"][:, : scores.shape[1]] & score_mask
            loss = soft_label_cross_entropy(scores, batch["cmax"][:, : scores.shape[1]], mask, temperature)
            losses.append(float(loss.detach().cpu()))
            correct += int((scores.masked_fill(~mask, -1e9).argmax(dim=1) == batch["labels"]).sum().detach().cpu())
            total += len(batch["labels"])
    return sum(losses) / max(1, len(losses)), correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--loss_type", choices=["soft_ce"], default="soft_ce")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--data_dir", default="data/ranker_dataset/")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
