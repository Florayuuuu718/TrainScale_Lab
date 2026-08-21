"""Typed TOML configuration for reproducible M1 experiments."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

DatasetName = Literal["synthetic", "cifar10"]
ModelName = Literal["mlp", "small_cnn"]
PrecisionName = Literal["fp32", "amp"]
DeviceName = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str = "synthetic_fp32"
    dataset: DatasetName = "synthetic"
    model: ModelName = "mlp"
    data_root: str = "01_pytorch_training/data"
    download: bool = False
    train_samples: int = 400
    valid_samples: int = 112
    input_dim: int = 16
    num_classes: int = 4
    hidden_dim: int = 32
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0
    accumulation_steps: int = 1
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.9
    precision: PrecisionName = "fp32"
    compile_model: bool = False
    device: DeviceName = "cpu"
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 7
    output_dir: str = "01_pytorch_training/results/raw/synthetic_fp32"
    resume: str | None = None

    def validate(self) -> None:
        if self.dataset == "synthetic" and self.model != "mlp":
            raise ValueError("synthetic dataset requires model='mlp'")
        if self.dataset == "cifar10" and self.model != "small_cnn":
            raise ValueError("CIFAR-10 requires model='small_cnn'")
        if self.dataset == "cifar10" and self.num_classes != 10:
            raise ValueError("CIFAR-10 requires num_classes=10")
        if self.precision == "amp" and self.device != "cuda":
            raise ValueError("AMP baseline requires device='cuda'")
        integer_fields = {
            "train_samples": self.train_samples,
            "valid_samples": self.valid_samples,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "scheduler_step_size": self.scheduler_step_size,
        }
        invalid = [name for name, value in integer_fields.items() if value <= 0]
        if invalid:
            raise ValueError(f"positive values required for: {', '.join(invalid)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **overrides: Any) -> ExperimentConfig:
        filtered = {name: value for name, value in overrides.items() if value is not None}
        config = replace(self, **filtered)
        config.validate()
        return config


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    with source.open("rb") as handle:
        values = tomllib.load(handle)
    unknown = set(values) - set(ExperimentConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown config fields: {', '.join(sorted(unknown))}")
    config = ExperimentConfig(**values)
    config.validate()
    return config
