"""Strict local and GPU experiment configuration for Module 07."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def _load(path: Path, table: str) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if set(payload) != {table} or not isinstance(payload[table], dict):
        raise ValueError(f"{path}: expected only a [{table}] table")
    return dict(payload[table])


def _positive_int(raw: dict[str, Any], key: str) -> None:
    if not isinstance(raw[key], int) or raw[key] <= 0:
        raise ValueError(f"{key} must be a positive integer")


def load_local_config(path: Path) -> dict[str, Any]:
    raw = _load(path, "local")
    expected = {
        "world_sizes",
        "cases",
        "batch_size",
        "input_dim",
        "hidden_dim",
        "output_dim",
        "sequence_length",
        "d_model",
        "num_heads",
        "learning_rate",
        "seed",
        "atol",
        "timeout_seconds",
    }
    if set(raw) != expected:
        raise ValueError(f"local fields must be exactly {sorted(expected)}")
    if raw["world_sizes"] != [2, 4]:
        raise ValueError("local TP correctness requires world sizes 2 and 4")
    if set(raw["cases"]) != {"tp_mlp", "tp_attention"}:
        raise ValueError("local cases must contain MLP and attention TP")
    for key in (
        "batch_size",
        "input_dim",
        "hidden_dim",
        "output_dim",
        "sequence_length",
        "d_model",
        "num_heads",
        "seed",
        "timeout_seconds",
    ):
        _positive_int(raw, key)
    for world_size in raw["world_sizes"]:
        if raw["hidden_dim"] % world_size or raw["num_heads"] % world_size:
            raise ValueError("hidden_dim and num_heads must divide every world size")
    if raw["d_model"] % raw["num_heads"]:
        raise ValueError("d_model must be divisible by num_heads")
    for key in ("learning_rate", "atol"):
        if not isinstance(raw[key], int | float) or raw[key] <= 0:
            raise ValueError(f"{key} must be positive")
    return raw


def load_gpu_config(path: Path) -> dict[str, Any]:
    raw = _load(path, "benchmark")
    expected = {
        "world_sizes",
        "strategies",
        "model_presets",
        "fsdp_wrap",
        "per_rank_batch_size",
        "warmup_steps",
        "measured_steps",
        "repetitions",
        "learning_rate",
        "seed",
        "timeout_seconds",
        "oom_safety_fraction",
        "enable_2d",
    }
    if set(raw) != expected:
        raise ValueError(f"benchmark fields must be exactly {sorted(expected)}")
    if raw["world_sizes"] != [2, 4]:
        raise ValueError("GPU benchmark requires world sizes 2 and 4")
    if set(raw["strategies"]) != {"ddp", "fsdp2", "tp"}:
        raise ValueError("GPU strategies must be DDP, FSDP2, and TP")
    if set(raw["model_presets"]) != {"small", "medium"}:
        raise ValueError("GPU benchmark requires small and medium presets")
    if set(raw["fsdp_wrap"]) != {"root", "layer"}:
        raise ValueError("FSDP wrap must compare root and layer")
    for key in (
        "per_rank_batch_size",
        "warmup_steps",
        "measured_steps",
        "repetitions",
        "seed",
        "timeout_seconds",
    ):
        _positive_int(raw, key)
    if not isinstance(raw["learning_rate"], int | float) or raw["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    if not isinstance(raw["oom_safety_fraction"], float) or not 0 < raw["oom_safety_fraction"] < 1:
        raise ValueError("oom_safety_fraction must be between zero and one")
    if not isinstance(raw["enable_2d"], bool):
        raise ValueError("enable_2d must be boolean")
    return raw
