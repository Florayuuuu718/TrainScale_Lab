"""Strict configuration contracts for TinyCollective experiments."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

DTYPES = {"float32", "float16", "bfloat16"}
ALGORITHMS = {"centralized", "ring", "torch"}


def _table(path: Path, name: str) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if set(payload) != {name} or not isinstance(payload[name], dict):
        raise ValueError(f"{path}: expected only a [{name}] table")
    return dict(payload[name])


def load_correctness_config(path: Path) -> dict[str, Any]:
    raw = _table(path, "correctness")
    expected = {
        "world_sizes",
        "element_counts",
        "algorithms",
        "dtype",
        "seed",
        "atol",
        "rtol",
        "timeout_seconds",
    }
    if set(raw) != expected:
        raise ValueError(f"{path}: fields must be exactly {sorted(expected)}")
    if raw["world_sizes"] != sorted(set(raw["world_sizes"])) or any(
        not isinstance(value, int) or value < 2 for value in raw["world_sizes"]
    ):
        raise ValueError("world_sizes must be sorted unique integers >= 2")
    if any(not isinstance(value, int) or value <= 0 for value in raw["element_counts"]):
        raise ValueError("element_counts must contain positive integers")
    if not set(raw["algorithms"]) <= {"centralized", "ring"} or not raw["algorithms"]:
        raise ValueError("correctness algorithms must contain centralized and/or ring")
    _validate_common(raw)
    return raw


def load_benchmark_config(path: Path) -> dict[str, Any]:
    raw = _table(path, "benchmark")
    expected = {
        "world_sizes",
        "message_bytes",
        "algorithms",
        "dtype",
        "warmup_iterations",
        "measured_iterations",
        "repetitions",
        "seed",
        "atol",
        "rtol",
        "timeout_seconds",
    }
    if set(raw) != expected:
        raise ValueError(f"{path}: fields must be exactly {sorted(expected)}")
    if raw["world_sizes"] != sorted(set(raw["world_sizes"])) or any(
        not isinstance(value, int) or value < 2 for value in raw["world_sizes"]
    ):
        raise ValueError("world_sizes must be sorted unique integers >= 2")
    if any(not isinstance(value, int) or value <= 0 for value in raw["message_bytes"]):
        raise ValueError("message_bytes must contain positive integers")
    if set(raw["algorithms"]) != {"centralized", "ring", "torch"}:
        raise ValueError("GPU benchmark must compare exactly centralized, ring, and torch")
    for key in ("warmup_iterations", "measured_iterations", "repetitions"):
        if not isinstance(raw[key], int) or raw[key] <= 0:
            raise ValueError(f"{key} must be positive")
    _validate_common(raw)
    return raw


def _validate_common(raw: dict[str, Any]) -> None:
    if raw["dtype"] not in DTYPES:
        raise ValueError(f"unsupported dtype: {raw['dtype']}")
    if not isinstance(raw["seed"], int) or raw["seed"] <= 0:
        raise ValueError("seed must be positive")
    for key in ("atol", "rtol"):
        if not isinstance(raw[key], int | float) or not math.isfinite(raw[key]) or raw[key] < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    if not isinstance(raw["timeout_seconds"], int) or raw["timeout_seconds"] <= 0:
        raise ValueError("timeout_seconds must be positive")
