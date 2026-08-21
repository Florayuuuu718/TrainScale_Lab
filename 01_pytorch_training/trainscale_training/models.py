"""Small transparent models for synthetic classification and CIFAR-10."""

from __future__ import annotations

import torch
from torch import nn

from .config import ExperimentConfig


def make_mlp(input_dim: int = 16, hidden_dim: int = 32, num_classes: int = 4) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )


class SmallCifarCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


def make_model(config: ExperimentConfig) -> nn.Module:
    if config.model == "mlp":
        return make_mlp(config.input_dim, config.hidden_dim, config.num_classes)
    return SmallCifarCNN(config.num_classes)
