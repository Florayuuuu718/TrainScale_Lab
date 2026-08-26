"""CPU-only configuration, sharding, and scaling contracts for module 03."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

CORRECTNESS_KEYS = {
    "seed",
    "dataset_size",
    "input_dim",
    "hidden_dim",
    "num_classes",
    "global_batch_size",
    "epochs",
    "learning_rate",
}
BENCHMARK_KEYS = {
    "device",
    "backend",
    "world_sizes",
    "modes",
    "seed",
    "input_dim",
    "hidden_dim",
    "num_classes",
    "global_batch_size",
    "per_rank_batch_size",
    "warmup_steps",
    "measured_steps",
    "learning_rate",
}


def _read_single_table(path: Path, table: str) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if set(payload) != {table} or not isinstance(payload[table], dict):
        raise ValueError(f"{path}: expected exactly one [{table}] table")
    return payload[table]


def _require_exact_keys(raw: dict[str, Any], expected: set[str], path: Path) -> None:
    missing = expected - set(raw)
    unknown = set(raw) - expected
    if missing or unknown:
        raise ValueError(f"{path}: missing={sorted(missing)} unknown={sorted(unknown)}")


def _validate_positive_ints(raw: dict[str, Any], keys: set[str], path: Path) -> None:
    invalid = [key for key in keys if not isinstance(raw[key], int) or raw[key] <= 0]
    if invalid:
        raise ValueError(f"{path}: positive integers required for {sorted(invalid)}")


def load_correctness_config(path: Path) -> dict[str, Any]:
    raw = _read_single_table(path, "experiment")
    _require_exact_keys(raw, CORRECTNESS_KEYS, path)
    _validate_positive_ints(raw, CORRECTNESS_KEYS - {"learning_rate"}, path)
    if not isinstance(raw["learning_rate"], (int, float)) or raw["learning_rate"] <= 0:
        raise ValueError(f"{path}: learning_rate must be positive")
    if raw["dataset_size"] % raw["global_batch_size"] != 0:
        raise ValueError(f"{path}: dataset_size must be divisible by global_batch_size")
    return dict(raw)


def load_benchmark_config(path: Path) -> dict[str, Any]:
    raw = _read_single_table(path, "benchmark")
    _require_exact_keys(raw, BENCHMARK_KEYS, path)
    _validate_positive_ints(
        raw,
        {
            "seed",
            "input_dim",
            "hidden_dim",
            "num_classes",
            "global_batch_size",
            "per_rank_batch_size",
            "measured_steps",
        },
        path,
    )
    if not isinstance(raw["warmup_steps"], int) or raw["warmup_steps"] < 0:
        raise ValueError(f"{path}: warmup_steps must be non-negative")
    if raw["device"] not in {"cpu", "cuda"}:
        raise ValueError(f"{path}: device must be cpu or cuda")
    if raw["backend"] not in {"gloo", "nccl"}:
        raise ValueError(f"{path}: backend must be gloo or nccl")
    if (raw["device"], raw["backend"]) not in {("cpu", "gloo"), ("cuda", "nccl")}:
        raise ValueError(f"{path}: use cpu/gloo or cuda/nccl")
    world_sizes = raw["world_sizes"]
    if (
        not isinstance(world_sizes, list)
        or not world_sizes
        or any(not isinstance(value, int) or value <= 0 for value in world_sizes)
        or len(world_sizes) != len(set(world_sizes))
    ):
        raise ValueError(f"{path}: world_sizes must be unique positive integers")
    if world_sizes != sorted(world_sizes) or world_sizes[0] != 1:
        raise ValueError(f"{path}: world_sizes must be sorted and start at 1")
    modes = raw["modes"]
    if not isinstance(modes, list) or not modes or set(modes) - {"strong", "weak"}:
        raise ValueError(f"{path}: modes must contain strong and/or weak")
    if len(modes) != len(set(modes)):
        raise ValueError(f"{path}: modes must not contain duplicates")
    if not isinstance(raw["learning_rate"], (int, float)) or raw["learning_rate"] <= 0:
        raise ValueError(f"{path}: learning_rate must be positive")
    for world_size in world_sizes:
        if raw["global_batch_size"] % world_size != 0:
            raise ValueError(
                f"{path}: global_batch_size must be divisible by world_size={world_size}"
            )
    return dict(raw)


def analyze_sampler_shards(shards: list[list[int]], dataset_size: int) -> dict[str, Any]:
    flattened = [index for shard in shards for index in shard]
    unique = set(flattened)
    expected_total = math.ceil(dataset_size / len(shards)) * len(shards)
    return {
        "rank_count": len(shards),
        "samples_per_rank": [len(shard) for shard in shards],
        "total_samples": len(flattened),
        "expected_total_samples": expected_total,
        "unique_samples": len(unique),
        "missing_samples": sorted(set(range(dataset_size)) - unique),
        "padding_duplicates": len(flattened) - len(unique),
        "coverage_complete": unique == set(range(dataset_size)),
    }


def add_scaling_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [record for record in records if record.get("status") == "success"]
    baselines = {
        record["mode"]: record
        for record in successful
        if record["world_size"] == 1
    }
    for record in records:
        if record.get("status") != "success":
            continue
        baseline = baselines.get(record["mode"])
        if baseline is None:
            record["speedup_over_1"] = None
            record["scaling_efficiency"] = None
            continue
        speedup = record["global_samples_per_second"] / baseline["global_samples_per_second"]
        record["speedup_over_1"] = speedup
        record["scaling_efficiency"] = speedup / record["world_size"]
    return records

