"""Versioned, atomic checkpoint save and exact training-state restoration."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

SCHEMA_VERSION = 1


def capture_rng_state(data_generator: torch.Generator | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if data_generator is not None:
        state["data_generator"] = data_generator.get_state()
    return state


def restore_rng_state(state: dict[str, Any], data_generator: torch.Generator | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if data_generator is not None and "data_generator" in state:
        data_generator.set_state(state["data_generator"])


def build_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    scheduler: Any = None,
    scaler: Any = None,
    data_generator: torch.Generator | None = None,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": capture_rng_state(data_generator),
        "config": config or {},
        "metrics": metrics or {},
        "metadata": {"torch_version": torch.__version__},
    }


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    scaler: Any = None,
    data_generator: torch.Generator | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {state.get('schema_version')}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    restore_rng_state(state["rng"], data_generator)
    return state
