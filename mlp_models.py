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


def save_checkpoint(path, model: CandidateScorer, input_dim: int, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(path, device: str = "cpu") -> CandidateScorer:
    checkpoint = torch.load(path, map_location=device)
    model = CandidateScorer(int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
