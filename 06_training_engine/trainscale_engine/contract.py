"""Strict local and GPU experiment configuration for Module 06."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

STRATEGIES = {"bulk", "per_parameter", "bucket_sync", "bucket_async", "ddp"}


def _load(path: Path, table: str) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if set(payload) != {table} or not isinstance(payload[table], dict):
        raise ValueError(f"{path}: expected only a [{table}] table")
    return dict(payload[table])


def _positive_ints(values: Any, name: str) -> None:
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, int) or value <= 0 for value in values)
    ):
        raise ValueError(f"{name} must contain positive integers")


def load_correctness_config(path: Path) -> dict[str, Any]:
    raw = _load(path, "correctness")
    expected = {
        "world_sizes",
        "strategies",
        "accumulation_steps",
        "model_preset",
        "global_batch_size",
        "bucket_cap_bytes",
        "include_unused",
        "learning_rate",
        "seed",
        "atol",
        "rtol",
        "timeout_seconds",
    }
    if set(raw) != expected:
        raise ValueError(f"correctness fields must be exactly {sorted(expected)}")
    _positive_ints(raw["world_sizes"], "world_sizes")
    _positive_ints(raw["accumulation_steps"], "accumulation_steps")
    if set(raw["strategies"]) != STRATEGIES:
        raise ValueError("correctness must cover every reducer strategy and DDP")
    if raw["model_preset"] != "small":
        raise ValueError("local correctness uses the small deterministic preset")
    if not isinstance(raw["include_unused"], bool):
        raise ValueError("include_unused must be boolean")
    for key in ("global_batch_size", "bucket_cap_bytes", "seed", "timeout_seconds"):
        if not isinstance(raw[key], int) or raw[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for world_size in raw["world_sizes"]:
        if raw["global_batch_size"] % world_size:
            raise ValueError("global_batch_size must divide every world size")
        local_batch = raw["global_batch_size"] // world_size
        if any(local_batch % steps for steps in raw["accumulation_steps"]):
            raise ValueError("local batch must divide every accumulation_steps value")
    for key in ("learning_rate", "atol", "rtol"):
        if not isinstance(raw[key], int | float) or raw[key] < 0:
            raise ValueError(f"{key} must be non-negative")
    return raw


def load_baseline_config(path: Path) -> dict[str, Any]:
    raw = _load(path, "baseline")
    expected = {
        "model_preset",
        "batch_size",
        "steps",
        "learning_rate",
        "seed",
        "device",
    }
    if set(raw) != expected:
        raise ValueError(f"baseline fields must be exactly {sorted(expected)}")
    if raw["model_preset"] not in {"small", "medium"}:
        raise ValueError("baseline model_preset must be small or medium")
    for key in ("batch_size", "steps", "seed"):
        if not isinstance(raw[key], int) or raw[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not isinstance(raw["learning_rate"], int | float) or raw["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    if raw["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    return raw


def load_benchmark_config(path: Path) -> dict[str, Any]:
    raw = _load(path, "benchmark")
    expected = {
        "world_sizes",
        "strategies",
        "model_presets",
        "bucket_cap_mb",
        "precisions",
        "accumulation_steps",
        "per_rank_batch_size",
        "warmup_steps",
        "measured_steps",
        "repetitions",
        "learning_rate",
        "seed",
        "timeout_seconds",
    }
    if set(raw) != expected:
        raise ValueError(f"benchmark fields must be exactly {sorted(expected)}")
    _positive_ints(raw["world_sizes"], "world_sizes")
    _positive_ints(raw["accumulation_steps"], "accumulation_steps")
    if set(raw["strategies"]) != STRATEGIES:
        raise ValueError("benchmark must cover every strategy")
    if set(raw["model_presets"]) != {"small", "medium"}:
        raise ValueError("benchmark requires small and medium model presets")
    if set(raw["precisions"]) != {"fp32", "amp"}:
        raise ValueError("benchmark requires fp32 and amp")
    if (
        not isinstance(raw["bucket_cap_mb"], list)
        or len(raw["bucket_cap_mb"]) < 3
        or any(not isinstance(value, int | float) or value <= 0 for value in raw["bucket_cap_mb"])
    ):
        raise ValueError("bucket_cap_mb requires at least three positive values")
    for key in (
        "per_rank_batch_size",
        "warmup_steps",
        "measured_steps",
        "repetitions",
        "seed",
        "timeout_seconds",
    ):
        if not isinstance(raw[key], int) or raw[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not isinstance(raw["learning_rate"], int | float) or raw["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    return raw
