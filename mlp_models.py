"""Small MLP scorer models shared by Stage C training and evaluation."""

from __future__ import annotations

import torch
from torch import nn


class CandidateScorer(nn.Module):
    """Scores each candidate independently from its feature vector."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [batch, candidates, feature_dim]
        batch, candidates, dim = features.shape
        scores = self.net(features.reshape(batch * candidates, dim)).reshape(batch, candidates)
        return scores


class ImprovementAwareCandidateScorer(nn.Module):
    """Scores candidates and predicts whether deviating from FIFO is beneficial."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.candidate_score_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.improvement_gate_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Keep the single-head scorer interface for existing evaluation helpers.
        batch, candidates, dim = features.shape
        return self.candidate_score_head(features.reshape(batch * candidates, dim)).reshape(batch, candidates)

    def gate_logits(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            pooled = features.mean(dim=1)
        else:
            weights = mask.to(features.dtype).unsqueeze(-1)
            pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.improvement_gate_head(pooled).squeeze(-1)

    def forward_with_gate(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self(features), self.gate_logits(features, mask)


def save_checkpoint(path, model: nn.Module, input_dim: int, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(path, device: str = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location=device)
    metadata = checkpoint.get("metadata") or {}
    model_type = metadata.get("type", "mlp_ranker")
    if model_type == "improvement_aware_ranker":
        model = ImprovementAwareCandidateScorer(int(checkpoint["input_dim"]))
    else:
        model = CandidateScorer(int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
