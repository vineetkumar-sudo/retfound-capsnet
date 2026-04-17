"""Baseline models for Day 1: linear probe, MLP classifier, MLP regressor."""

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """Single linear layer: 1024 -> num_classes."""

    def __init__(self, feature_dim: int = 1024, num_classes: int = 5):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPClassifier(nn.Module):
    """MLP with ReLU + dropout: 1024 -> hidden -> num_classes."""

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dims: list[int] | None = None,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        layers = []
        in_dim = feature_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPRegressor(nn.Module):
    """MLP with single output for ordinal regression (MSE on grade)."""

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        layers = []
        in_dim = feature_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
