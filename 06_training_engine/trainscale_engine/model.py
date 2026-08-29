"""Small deterministic Transformer workload shared by Modules 06 and 07."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

MODEL_PRESETS = {
    "small": {
        "vocabulary_size": 128,
        "sequence_length": 8,
        "d_model": 16,
        "nhead": 4,
        "feedforward_dim": 32,
        "layers": 1,
        "num_classes": 4,
    },
    "medium": {
        "vocabulary_size": 512,
        "sequence_length": 32,
        "d_model": 128,
        "nhead": 8,
        "feedforward_dim": 512,
        "layers": 4,
        "num_classes": 32,
    },
}


@dataclass(frozen=True)
class TinyTransformerConfig:
    vocabulary_size: int
    sequence_length: int
    d_model: int
    nhead: int
    feedforward_dim: int
    layers: int
    num_classes: int

    def validate(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")


class TinyTransformer(nn.Module):
    """Encoder-only classifier with no dropout, suitable for exact comparisons."""

    def __init__(self, config: TinyTransformerConfig, *, include_unused: bool = False) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.d_model)
        self.position_embedding = nn.Parameter(torch.empty(config.sequence_length, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.feedforward_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        if include_unused:
            self.unused_probe = nn.Parameter(torch.zeros(config.d_model))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != self.config.sequence_length:
            raise ValueError(f"expected tokens shaped [batch, {self.config.sequence_length}]")
        hidden = self.token_embedding(tokens) + self.position_embedding.unsqueeze(0)
        hidden = self.encoder(hidden)
        return self.classifier(self.final_norm(hidden.mean(dim=1)))


def make_classification_batch(
    batch_size: int, config: TinyTransformerConfig, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        config.vocabulary_size,
        (batch_size, config.sequence_length),
        generator=generator,
    )
    labels = (tokens.sum(dim=1) % config.num_classes).long()
    return tokens, labels


def model_preset(name: str) -> TinyTransformerConfig:
    if name not in MODEL_PRESETS:
        raise ValueError(f"unknown model preset: {name}")
    return TinyTransformerConfig(**MODEL_PRESETS[name])
