"""Synthetic and CIFAR-10 datasets with deterministic sampling boundaries."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split

from .config import ExperimentConfig


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    valid_loader: DataLoader
    generator: torch.Generator


def make_classification_dataset(
    num_samples: int = 512,
    input_dim: int = 16,
    num_classes: int = 4,
    seed: int = 0,
) -> TensorDataset:
    """Generate features and labels from a hidden learnable linear rule."""
    if min(num_samples, input_dim, num_classes) <= 0:
        raise ValueError("dataset dimensions must be positive")
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(num_samples, input_dim, generator=generator)
    teacher = torch.randn(input_dim, num_classes, generator=generator)
    targets = (features @ teacher).argmax(dim=1)
    return TensorDataset(features, targets)


class DelayedSyntheticDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Synthetic samples with optional per-item latency for worker experiments."""

    def __init__(self, num_samples: int, input_dim: int, delay_ms: float, seed: int = 0):
        self.dataset = make_classification_dataset(num_samples, input_dim, 4, seed)
        self.delay_seconds = delay_ms / 1000.0

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return self.dataset[index]


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _subset(dataset: Dataset, size: int, seed: int) -> Subset:
    dataset_size = len(dataset)  # type: ignore[arg-type]
    if size > dataset_size:
        raise ValueError(f"requested {size} samples from a dataset of size {dataset_size}")
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(seed))[:size]
    return Subset(dataset, indices.tolist())


def _make_cifar10(config: ExperimentConfig) -> tuple[Dataset, Dataset]:
    from torchvision import datasets, transforms  # type: ignore[import-untyped]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    root = Path(config.data_root)
    train = datasets.CIFAR10(root, train=True, transform=transform, download=config.download)
    valid = datasets.CIFAR10(root, train=False, transform=transform, download=config.download)
    return _subset(train, config.train_samples, config.seed), _subset(
        valid, config.valid_samples, config.seed + 1
    )


def make_data(config: ExperimentConfig) -> DataBundle:
    generator = torch.Generator().manual_seed(config.seed)
    train: Dataset
    valid: Dataset
    if config.dataset == "synthetic":
        full = make_classification_dataset(
            config.train_samples + config.valid_samples,
            config.input_dim,
            config.num_classes,
            config.seed,
        )
        train, valid = random_split(
            full, [config.train_samples, config.valid_samples], generator=generator
        )
    else:
        train, valid = _make_cifar10(config)

    train_loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=config.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=config.num_workers > 0,
    )
    return DataBundle(train_loader, valid_loader, generator)
