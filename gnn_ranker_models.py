"""Lightweight pure-PyTorch GNN-Ranker for Stage D."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
from torch import nn


class BipartiteGNNRanker(nn.Module):
    """Scores candidate jobs with two rounds of job-machine message passing."""

    def __init__(
        self,
        job_feature_dim: int,
        machine_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.job_feature_dim = int(job_feature_dim)
        self.machine_feature_dim = int(machine_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        self.job_encoder = nn.Linear(job_feature_dim, hidden_dim)
        self.machine_encoder = nn.Linear(machine_feature_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)
        self.job_to_machine = nn.ModuleList(
            nn.Linear(hidden_dim * 3, hidden_dim) for _ in range(num_layers)
        )
        self.machine_to_job = nn.ModuleList(
            nn.Linear(hidden_dim * 3, hidden_dim) for _ in range(num_layers)
        )
        self.job_update = nn.ModuleList(nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers))
        self.machine_update = nn.ModuleList(nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers))
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.activation = nn.ReLU()

    def forward_graph(self, graph: dict) -> torch.Tensor:
        job_x = graph["job_features"]
        machine_x = graph["machine_features"]
        edge_index = graph["edge_index"].long()
        edge_features = graph["edge_features"]
        candidate_indices = graph["candidate_job_indices"].long()

        job_h = self.activation(self.job_encoder(job_x))
        machine_h = self.activation(self.machine_encoder(machine_x))
        edge_h = self.activation(self.edge_encoder(edge_features))

        if edge_index.numel() > 0:
            src_job = edge_index[0]
            dst_machine = edge_index[1]
            for layer_idx in range(self.num_layers):
                jm_input = torch.cat([job_h[src_job], machine_h[dst_machine], edge_h], dim=1)
                jm_msg = self.activation(self.job_to_machine[layer_idx](jm_input))
                machine_agg = _mean_aggregate(jm_msg, dst_machine, machine_h.shape[0])
                machine_h = self.activation(self.machine_update[layer_idx](torch.cat([machine_h, machine_agg], dim=1)))

                mj_input = torch.cat([machine_h[dst_machine], job_h[src_job], edge_h], dim=1)
                mj_msg = self.activation(self.machine_to_job[layer_idx](mj_input))
                job_agg = _mean_aggregate(mj_msg, src_job, job_h.shape[0])
                job_h = self.activation(self.job_update[layer_idx](torch.cat([job_h, job_agg], dim=1)))

        return self.scorer(job_h[candidate_indices]).squeeze(-1)

    def forward(self, graphs: Iterable[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        scores: List[torch.Tensor] = [self.forward_graph(graph) for graph in graphs]
        max_candidates = max(score.numel() for score in scores)
        padded = scores[0].new_full((len(scores), max_candidates), -1e9)
        mask = torch.zeros((len(scores), max_candidates), dtype=torch.bool, device=padded.device)
        for row_idx, score in enumerate(scores):
            padded[row_idx, : score.numel()] = score
            mask[row_idx, : score.numel()] = True
        return padded, mask


def _mean_aggregate(messages: torch.Tensor, index: torch.Tensor, output_size: int) -> torch.Tensor:
    out = messages.new_zeros((output_size, messages.shape[1]))
    counts = messages.new_zeros((output_size, 1))
    out.index_add_(0, index, messages)
    counts.index_add_(0, index, torch.ones((messages.shape[0], 1), dtype=messages.dtype, device=messages.device))
    return out / counts.clamp_min(1.0)


def graph_to_torch(graph, device: str | torch.device = "cpu") -> dict:
    return {
        "job_features": torch.tensor(graph.job_features, dtype=torch.float32, device=device),
        "machine_features": torch.tensor(graph.machine_features, dtype=torch.float32, device=device),
        "edge_index": torch.tensor(graph.edge_index, dtype=torch.long, device=device),
        "edge_features": torch.tensor(graph.edge_features, dtype=torch.float32, device=device),
        "candidate_job_indices": torch.tensor(graph.candidate_job_indices, dtype=torch.long, device=device),
    }


def save_checkpoint(path, model: BipartiteGNNRanker, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "job_feature_dim": model.job_feature_dim,
            "machine_feature_dim": model.machine_feature_dim,
            "edge_feature_dim": model.edge_feature_dim,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(path, device: str = "cpu") -> BipartiteGNNRanker:
    checkpoint = torch.load(path, map_location=device)
    model = BipartiteGNNRanker(
        int(checkpoint["job_feature_dim"]),
        int(checkpoint["machine_feature_dim"]),
        int(checkpoint["edge_feature_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        num_layers=int(checkpoint.get("num_layers", 2)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
