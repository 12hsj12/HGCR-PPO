"""Lightweight pure-PyTorch GNN-Ranker for Stage D."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
from torch import nn


GNN_VARIANTS = ["gnn_v1", "gnn_v2_edge_fusion", "gnn_v2_no_fusion", "gnn_residual"]


class BipartiteGNNRanker(nn.Module):
    """Scores candidate jobs with edge-aware message passing and optional feature fusion."""

    def __init__(
        self,
        job_feature_dim: int,
        machine_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        candidate_feature_dim: int = 0,
        use_candidate_feature_fusion: bool = True,
        gnn_variant: str = "gnn_v2_edge_fusion",
        residual_alpha: float = 0.5,
    ):
        super().__init__()
        if gnn_variant not in GNN_VARIANTS:
            raise ValueError(f"Unknown gnn_variant {gnn_variant!r}. Expected one of {GNN_VARIANTS}.")
        self.job_feature_dim = int(job_feature_dim)
        self.machine_feature_dim = int(machine_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.gnn_variant = gnn_variant
        self.residual_alpha_value = float(residual_alpha)
        self.use_candidate_feature_fusion = bool(
            gnn_variant == "gnn_v2_edge_fusion" and use_candidate_feature_fusion and candidate_feature_dim > 0
        )
        self.use_residual = gnn_variant == "gnn_residual"

        self.job_encoder = nn.Linear(job_feature_dim, hidden_dim)
        self.machine_encoder = nn.Linear(machine_feature_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)
        self.edge_mlp = nn.ModuleList(
            _mlp(hidden_dim * 3, hidden_dim, hidden_dim, dropout) for _ in range(num_layers)
        )
        self.machine_update_mlp = nn.ModuleList(
            _mlp(hidden_dim * 2, hidden_dim, hidden_dim, dropout) for _ in range(num_layers)
        )
        self.job_update_mlp = nn.ModuleList(
            _mlp(hidden_dim * 2, hidden_dim, hidden_dim, dropout) for _ in range(num_layers)
        )
        if self.use_candidate_feature_fusion:
            self.candidate_feature_encoder = _mlp(candidate_feature_dim, hidden_dim, hidden_dim, dropout)
            scorer_dim = hidden_dim * 2
        else:
            self.candidate_feature_encoder = None
            scorer_dim = hidden_dim
        if self.use_residual:
            if candidate_feature_dim <= 0:
                raise ValueError("gnn_residual requires candidate_feature_dim > 0.")
            self.mlp_score_head = nn.Sequential(
                nn.Linear(candidate_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.gnn_delta_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.residual_alpha = nn.Parameter(torch.tensor(float(residual_alpha), dtype=torch.float32))
        else:
            self.mlp_score_head = None
            self.gnn_delta_head = None
            self.register_buffer("residual_alpha", torch.tensor(float(residual_alpha), dtype=torch.float32))
        self.scorer = nn.Sequential(
            nn.Linear(scorer_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward_graph(self, graph: dict) -> torch.Tensor:
        job_x = graph["job_features"]
        machine_x = graph["machine_features"]
        edge_index = graph["edge_index"].long()
        edge_features = graph["edge_features"]
        candidate_indices = graph["candidate_job_indices"].long()

        job_h = self.dropout(self.activation(self.job_encoder(job_x)))
        machine_h = self.dropout(self.activation(self.machine_encoder(machine_x)))
        edge_h = self.dropout(self.activation(self.edge_encoder(edge_features)))

        if edge_index.numel() > 0:
            src_job = edge_index[0]
            dst_machine = edge_index[1]
            for layer_idx in range(self.num_layers):
                edge_input = torch.cat([job_h[src_job], machine_h[dst_machine], edge_h], dim=1)
                edge_msg = self.edge_mlp[layer_idx](edge_input)
                machine_agg = _mean_aggregate(edge_msg, dst_machine, machine_h.shape[0])
                machine_h = self.machine_update_mlp[layer_idx](torch.cat([machine_h, machine_agg], dim=1))

                refreshed_edge_input = torch.cat([job_h[src_job], machine_h[dst_machine], edge_h], dim=1)
                refreshed_edge_msg = self.edge_mlp[layer_idx](refreshed_edge_input)
                job_agg = _mean_aggregate(refreshed_edge_msg, src_job, job_h.shape[0])
                job_h = self.job_update_mlp[layer_idx](torch.cat([job_h, job_agg], dim=1))

        candidate_h = job_h[candidate_indices]
        if self.use_residual:
            if "candidate_features" not in graph:
                raise KeyError("candidate_features are required when gnn_residual is enabled.")
            mlp_score = self.mlp_score_head(graph["candidate_features"]).squeeze(-1)
            gnn_delta_score = self.gnn_delta_head(candidate_h).squeeze(-1)
            return mlp_score + self.residual_alpha * gnn_delta_score
        if self.use_candidate_feature_fusion:
            if "candidate_features" not in graph:
                raise KeyError("candidate_features are required when candidate feature fusion is enabled.")
            candidate_features = graph["candidate_features"]
            candidate_feature_h = self.candidate_feature_encoder(candidate_features)
            candidate_h = torch.cat([candidate_h, candidate_feature_h], dim=1)
        return self.scorer(candidate_h).squeeze(-1)

    def forward(self, graphs: Iterable[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        scores: List[torch.Tensor] = [self.forward_graph(graph) for graph in graphs]
        max_candidates = max(score.numel() for score in scores)
        padded = scores[0].new_full((len(scores), max_candidates), -1e9)
        mask = torch.zeros((len(scores), max_candidates), dtype=torch.bool, device=padded.device)
        for row_idx, score in enumerate(scores):
            padded[row_idx, : score.numel()] = score
            mask[row_idx, : score.numel()] = True
        return padded, mask


class LegacyBipartiteGNNRanker(nn.Module):
    """Stage D v1 architecture, kept so old checkpoints evaluate unchanged."""

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
        self.dropout_p = 0.0
        self.candidate_feature_dim = 0
        self.use_candidate_feature_fusion = False
        self.gnn_variant = "gnn_v1"
        self.residual_alpha_value = 0.0

        self.job_encoder = nn.Linear(job_feature_dim, hidden_dim)
        self.machine_encoder = nn.Linear(machine_feature_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)
        self.job_to_machine = nn.ModuleList(nn.Linear(hidden_dim * 3, hidden_dim) for _ in range(num_layers))
        self.machine_to_job = nn.ModuleList(nn.Linear(hidden_dim * 3, hidden_dim) for _ in range(num_layers))
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


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )


def graph_to_torch(graph, device: str | torch.device = "cpu", candidate_features=None) -> dict:
    out = {
        "job_features": torch.tensor(graph.job_features, dtype=torch.float32, device=device),
        "machine_features": torch.tensor(graph.machine_features, dtype=torch.float32, device=device),
        "edge_index": torch.tensor(graph.edge_index, dtype=torch.long, device=device),
        "edge_features": torch.tensor(graph.edge_features, dtype=torch.float32, device=device),
        "candidate_job_indices": torch.tensor(graph.candidate_job_indices, dtype=torch.long, device=device),
    }
    if candidate_features is not None:
        out["candidate_features"] = torch.tensor(candidate_features, dtype=torch.float32, device=device)
    return out


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
            "dropout": model.dropout_p,
            "candidate_feature_dim": model.candidate_feature_dim,
            "use_candidate_feature_fusion": model.use_candidate_feature_fusion,
            "gnn_variant": getattr(model, "gnn_variant", "gnn_v2_edge_fusion"),
            "residual_alpha": float(getattr(model, "residual_alpha", torch.tensor(0.5)).detach().cpu()),
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(path, device: str = "cpu") -> BipartiteGNNRanker:
    checkpoint = torch.load(path, map_location=device)
    checkpoint_variant = checkpoint.get("gnn_variant")
    if checkpoint_variant == "gnn_v1" or "candidate_feature_dim" not in checkpoint:
        model = LegacyBipartiteGNNRanker(
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
    model = BipartiteGNNRanker(
        int(checkpoint["job_feature_dim"]),
        int(checkpoint["machine_feature_dim"]),
        int(checkpoint["edge_feature_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        num_layers=int(checkpoint.get("num_layers", 2)),
        dropout=float(checkpoint.get("dropout", 0.0)),
        candidate_feature_dim=int(checkpoint.get("candidate_feature_dim", 0)),
        use_candidate_feature_fusion=bool(checkpoint.get("use_candidate_feature_fusion", False)),
        gnn_variant=str(checkpoint.get("gnn_variant", "gnn_v2_edge_fusion")),
        residual_alpha=float(checkpoint.get("residual_alpha", 0.5)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
