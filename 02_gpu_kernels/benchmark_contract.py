"""CPU-only configuration and result helpers for module 02 benchmarks."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

ALLOWED_DTYPES = {"float16", "bfloat16", "float32"}
ALLOWED_LAYOUTS = {"contiguous"}
ALLOWED_OPERATORS = {
    "attention",
    "layer_norm",
    "layer_norm_backward",
    "matmul",
    "relu_add",
    "softmax",
    "vector_add",
}
CASE_KEYS = {
    "id",
    "operator",
    "shape",
    "dtype",
    "layout",
    "inner",
    "causal",
    "eps",
}
MATMUL_CANDIDATE_KEYS = {
    "id",
    "block_m",
    "block_n",
    "block_k",
    "group_m",
    "num_warps",
}


def percentile(values: list[float], fraction: float) -> float:
    """Return the nearest-rank sample used by all module 02 reports."""
    if not values:
        raise ValueError("percentile requires at least one sample")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _validate_case(raw: dict[str, Any], *, source: Path) -> tuple[str, dict[str, Any]]:
    unknown = set(raw) - CASE_KEYS
    if unknown:
        raise ValueError(f"{source}: unknown case fields: {sorted(unknown)}")
    required = {"id", "operator", "shape", "dtype", "layout", "inner"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"{source}: missing case fields: {sorted(missing)}")

    case_id = raw["id"]
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"{source}: case id must be a non-empty string")
    operator = raw["operator"]
    if operator not in ALLOWED_OPERATORS:
        raise ValueError(f"{source}: unsupported operator {operator!r}")
    shape = raw["shape"]
    if not isinstance(shape, list) or not shape or any(
        not isinstance(value, int) or value <= 0 for value in shape
    ):
        raise ValueError(f"{source}: shape must contain positive integers")
    if raw["dtype"] not in ALLOWED_DTYPES:
        raise ValueError(f"{source}: unsupported dtype {raw['dtype']!r}")
    if raw["layout"] not in ALLOWED_LAYOUTS:
        raise ValueError(f"{source}: unsupported layout {raw['layout']!r}")
    if not isinstance(raw["inner"], int) or raw["inner"] <= 0:
        raise ValueError(f"{source}: inner must be a positive integer")
    if "causal" in raw and not isinstance(raw["causal"], bool):
        raise ValueError(f"{source}: causal must be true or false")
    if "eps" in raw and (
        not isinstance(raw["eps"], float | int)
        or not math.isfinite(raw["eps"])
        or raw["eps"] <= 0
    ):
        raise ValueError(f"{source}: eps must be a finite positive number")

    case = {key: value for key, value in raw.items() if key != "id"}
    return case_id, case


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    """Load and strictly validate a TOML file containing ``[[case]]`` tables."""
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if set(payload) != {"case"} or not isinstance(payload["case"], list):
        raise ValueError(f"{path}: expected only a [[case]] array")
    cases: dict[str, dict[str, Any]] = {}
    for raw in payload["case"]:
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: every case must be a TOML table")
        case_id, case = _validate_case(raw, source=path)
        if case_id in cases:
            raise ValueError(f"{path}: duplicate case id {case_id!r}")
        cases[case_id] = case
    if not cases:
        raise ValueError(f"{path}: at least one case is required")
    return cases


def load_case_ids(path: Path, available: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    """Load a named subset without duplicating the full case definitions."""
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if set(payload) != {"suite"} or set(payload["suite"]) != {"case_ids"}:
        raise ValueError(f"{path}: expected only [suite].case_ids")
    case_ids = payload["suite"]["case_ids"]
    if not isinstance(case_ids, list) or not case_ids:
        raise ValueError(f"{path}: case_ids must be a non-empty list")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{path}: duplicate case id in suite")
    missing = [case_id for case_id in case_ids if case_id not in available]
    if missing:
        raise ValueError(f"{path}: unknown case ids: {missing}")
    return tuple(case_ids)


def select_case_ids(
    available: dict[str, dict[str, Any]],
    suite_case_ids: tuple[str, ...],
    *,
    operators: tuple[str, ...] = (),
    requested_case_ids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Select a beginner-sized subset while preserving configuration order."""
    if operators and requested_case_ids:
        raise ValueError("choose operators or explicit case ids, not both")
    if requested_case_ids:
        if len(requested_case_ids) != len(set(requested_case_ids)):
            raise ValueError("requested case ids must not contain duplicates")
        unknown = [case_id for case_id in requested_case_ids if case_id not in available]
        if unknown:
            raise ValueError(f"unknown requested case ids: {unknown}")
        return requested_case_ids
    if operators:
        unknown_operators = sorted(set(operators) - ALLOWED_OPERATORS)
        if unknown_operators:
            raise ValueError(f"unknown requested operators: {unknown_operators}")
        selected = tuple(
            case_id
            for case_id, case in available.items()
            if case["operator"] in operators
        )
        if not selected:
            raise ValueError("operator selection did not match any configured case")
        return selected
    return suite_case_ids


def load_matmul_candidates(path: Path) -> dict[str, dict[str, int]]:
    """Load the finite tile/warp search space used by the teaching autotuner."""
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if set(payload) != {"candidate"} or not isinstance(payload["candidate"], list):
        raise ValueError(f"{path}: expected only a [[candidate]] array")
    candidates: dict[str, dict[str, int]] = {}
    for raw in payload["candidate"]:
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: every candidate must be a TOML table")
        unknown = set(raw) - MATMUL_CANDIDATE_KEYS
        missing = MATMUL_CANDIDATE_KEYS - set(raw)
        if unknown or missing:
            raise ValueError(
                f"{path}: candidate fields missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        candidate_id = raw["id"]
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"{path}: candidate id must be a non-empty string")
        integer_fields = MATMUL_CANDIDATE_KEYS - {"id"}
        if any(not isinstance(raw[key], int) or raw[key] <= 0 for key in integer_fields):
            raise ValueError(f"{path}: candidate values must be positive integers")
        if raw["num_warps"] not in {1, 2, 4, 8}:
            raise ValueError(f"{path}: num_warps must be 1, 2, 4, or 8")
        if any(raw[key] & (raw[key] - 1) for key in ("block_m", "block_n", "block_k")):
            raise ValueError(f"{path}: block sizes must be powers of two")
        if candidate_id in candidates:
            raise ValueError(f"{path}: duplicate candidate id {candidate_id!r}")
        candidates[candidate_id] = {
            key: raw[key] for key in integer_fields
        }
    if not candidates:
        raise ValueError(f"{path}: at least one candidate is required")
    return candidates


def validate_result_record(record: dict[str, Any]) -> None:
    """Validate the stable fields required for a successful benchmark record."""
    required = {"case_id", "implementation", "status", "correctness", "steady_state"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"result record is missing: {sorted(missing)}")
    if record["status"] != "success":
        raise ValueError("validate_result_record only accepts successful records")
    if record["correctness"].get("status") != "passed":
        raise ValueError("performance data requires a passed correctness gate")
    latency = record["steady_state"].get("latency_us", {})
    if set(latency) != {"median", "p10", "p90"}:
        raise ValueError("latency_us must contain median, p10, and p90")
    if any(not isinstance(latency[key], float | int) or latency[key] <= 0 for key in latency):
        raise ValueError("all latency values must be positive")
    if not latency["p10"] <= latency["median"] <= latency["p90"]:
        raise ValueError("latency percentiles are not ordered")
