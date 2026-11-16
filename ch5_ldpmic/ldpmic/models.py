"""Models.  All small enough to train on a laptop CPU."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Two-hidden-layer ReLU MLP.  Tabular default (64, 32); images (256, 128)
    on flattened pixels."""

    def __init__(self, in_dim: int, n_classes: int, hidden=(64, 32)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class TinyCNN(nn.Module):
    """Optional small CNN for 28x28 grayscale images (input flattened or not)."""

    def __init__(self, n_classes: int = 10, side: int = 28):
        super().__init__()
        self.side = side
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Linear(16 * (side // 4) ** 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], 1, self.side, self.side)
        return self.head(self.features(x).flatten(1))


def build_model(name: str, in_dim: int, n_classes: int) -> nn.Module:
    if name == "mlp_tabular":
        return MLP(in_dim, n_classes, hidden=(64, 32))
    if name == "mlp_image":
        return MLP(in_dim, n_classes, hidden=(256, 128))
    if name == "tiny_cnn":
        return TinyCNN(n_classes)
    raise ValueError(f"unknown model {name!r}")


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
