"""Strict TOML and command contracts for module 04 nccl-tests experiments."""

from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

COLLECTIVE_BINARIES = {
    "all_reduce": "all_reduce_perf",
    "all_gather": "all_gather_perf",
    "reduce_scatter": "reduce_scatter_perf",
    "broadcast": "broadcast_perf",
}
NCCL_DTYPES = {"float", "half", "bfloat16"}
CASE_KEYS = {
    "id",
    "collective",
    "devices",
    "min_bytes",
    "max_bytes",
    "step_factor",
    "warmup_iterations",
    "measured_iterations",
    "dtype",
}
BRIDGE_KEYS = {
    "world_sizes",
    "seed",
    "input_dim",
    "hidden_dim",
    "num_classes",
    "per_rank_batch_size",
    "warmup_steps",
    "profile_steps",
    "bucket_cap_mb",
    "learning_rate",
}


@dataclass(frozen=True)
class NcclCase:
    id: str
    collective: str
    devices: tuple[int, ...]
    min_bytes: int
    max_bytes: int
    step_factor: float
    warmup_iterations: int
    measured_iterations: int
    dtype: str

    @property
    def world_size(self) -> int:
        return len(self.devices)

    @property
    def binary(self) -> str:
        return COLLECTIVE_BINARIES[self.collective]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["devices"] = list(self.devices)
        payload["world_size"] = self.world_size
        payload["binary"] = self.binary
        return payload


def _validate_case(raw: dict[str, Any], source: Path) -> NcclCase:
    unknown = set(raw) - CASE_KEYS
    missing = CASE_KEYS - set(raw)
    if unknown or missing:
        raise ValueError(
            f"{source}: case fields missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    case_id = raw["id"]
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"{source}: id must be a non-empty string")
    if raw["collective"] not in COLLECTIVE_BINARIES:
        raise ValueError(f"{source}: unsupported collective {raw['collective']!r}")
    devices = raw["devices"]
    if (
        not isinstance(devices, list)
        or len(devices) < 2
        or any(not isinstance(device, int) or device < 0 for device in devices)
        or len(devices) != len(set(devices))
    ):
        raise ValueError(f"{source}: devices must contain at least two unique non-negative ids")
    for field in ("min_bytes", "max_bytes", "measured_iterations"):
        if not isinstance(raw[field], int) or raw[field] <= 0:
            raise ValueError(f"{source}: {field} must be a positive integer")
    if raw["min_bytes"] > raw["max_bytes"]:
        raise ValueError(f"{source}: min_bytes must not exceed max_bytes")
    if (
        not isinstance(raw["step_factor"], int | float)
        or not math.isfinite(raw["step_factor"])
        or raw["step_factor"] <= 1
    ):
        raise ValueError(f"{source}: step_factor must be finite and greater than one")
    if not isinstance(raw["warmup_iterations"], int) or raw["warmup_iterations"] < 0:
        raise ValueError(f"{source}: warmup_iterations must be a non-negative integer")
    if raw["dtype"] not in NCCL_DTYPES:
        raise ValueError(f"{source}: unsupported dtype {raw['dtype']!r}")
    return NcclCase(
        id=case_id,
        collective=raw["collective"],
        devices=tuple(devices),
        min_bytes=raw["min_bytes"],
        max_bytes=raw["max_bytes"],
        step_factor=float(raw["step_factor"]),
        warmup_iterations=raw["warmup_iterations"],
        measured_iterations=raw["measured_iterations"],
        dtype=raw["dtype"],
    )


def load_cases(path: Path) -> list[NcclCase]:
    """Load a strict ``[[case]]`` array while retaining declaration order."""
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if set(payload) != {"case"} or not isinstance(payload["case"], list):
        raise ValueError(f"{path}: expected only a [[case]] array")
    cases = [_validate_case(raw, path) for raw in payload["case"]]
    if not cases:
        raise ValueError(f"{path}: at least one case is required")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: case ids must be unique")
    return cases


def nccl_test_command(binary_directory: Path, case: NcclCase) -> list[str]:
    """Build the documented single-process, multi-GPU nccl-tests command."""
    return [
        str(binary_directory / case.binary),
        "-b",
        str(case.min_bytes),
        "-e",
        str(case.max_bytes),
        "-f",
        str(case.step_factor),
        "-g",
        str(case.world_size),
        "-w",
        str(case.warmup_iterations),
        "-n",
        str(case.measured_iterations),
        "-d",
        case.dtype,
    ]


def expected_bus_bandwidth(algbw_gbps: float, collective: str, world_size: int) -> float:
    """Return the nccl-tests bus-bandwidth normalization for one host."""
    if collective not in COLLECTIVE_BINARIES:
        raise ValueError(f"unsupported collective {collective!r}")
    if world_size < 2:
        raise ValueError("world_size must be at least two")
    if algbw_gbps < 0 or not math.isfinite(algbw_gbps):
        raise ValueError("algbw_gbps must be finite and non-negative")
    if collective == "all_reduce":
        factor = 2 * (world_size - 1) / world_size
    elif collective in {"all_gather", "reduce_scatter"}:
        factor = (world_size - 1) / world_size
    else:
        factor = 1.0
    return algbw_gbps * factor


def load_bridge_config(path: Path) -> dict[str, Any]:
    """Load the fixed DDP workload that connects module 03 and module 04."""
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if set(payload) != {"bridge"} or not isinstance(payload["bridge"], dict):
        raise ValueError(f"{path}: expected only a [bridge] table")
    raw = payload["bridge"]
    unknown = set(raw) - BRIDGE_KEYS
    missing = BRIDGE_KEYS - set(raw)
    if unknown or missing:
        raise ValueError(
            f"{path}: bridge fields missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    world_sizes = raw["world_sizes"]
    if (
        not isinstance(world_sizes, list)
        or not world_sizes
        or any(not isinstance(value, int) or value < 2 for value in world_sizes)
        or world_sizes != sorted(set(world_sizes))
    ):
        raise ValueError(f"{path}: world_sizes must be sorted unique integers >= 2")
    positive_ints = {
        "seed",
        "input_dim",
        "hidden_dim",
        "num_classes",
        "per_rank_batch_size",
        "profile_steps",
    }
    for key in positive_ints:
        if not isinstance(raw[key], int) or raw[key] <= 0:
            raise ValueError(f"{path}: {key} must be a positive integer")
    if not isinstance(raw["warmup_steps"], int) or raw["warmup_steps"] < 0:
        raise ValueError(f"{path}: warmup_steps must be a non-negative integer")
    for key in ("bucket_cap_mb", "learning_rate"):
        if (
            not isinstance(raw[key], int | float)
            or not math.isfinite(raw[key])
            or raw[key] <= 0
        ):
            raise ValueError(f"{path}: {key} must be finite and positive")
    return dict(raw)


def module03_mlp_parameter_count(input_dim: int, hidden_dim: int, num_classes: int) -> int:
    """Return parameters in module 03's Linear-ReLU-Linear workload."""
    if any(value <= 0 for value in (input_dim, hidden_dim, num_classes)):
        raise ValueError("model dimensions must be positive")
    return input_dim * hidden_dim + hidden_dim + hidden_dim * num_classes + num_classes

