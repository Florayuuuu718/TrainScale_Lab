"""Aggregate a five-run long-window DDP scaling follow-up with a stability gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

IDENTITY_KEYS = (
    "mode",
    "world_size",
    "backend",
    "device",
    "local_batch_size",
    "global_batch_size",
    "warmup_steps",
    "measured_steps",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_index(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("each DDP scaling source must contain a records list")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (str(record["mode"]), int(record["world_size"]))
        if key in indexed:
            raise ValueError(f"duplicate DDP scaling record: {key}")
        indexed[key] = record
    return indexed


def validate_sources(payloads: list[dict[str, Any]], expected_repetitions: int = 5) -> None:
    if len(payloads) != expected_repetitions:
        raise ValueError(
            f"long-window DDP scaling requires exactly {expected_repetitions} repetitions"
        )
    reference = payloads[0]
    config = reference.get("config", {})
    if config.get("world_sizes") != [1, 2, 4] or config.get("modes") != ["strong", "weak"]:
        raise ValueError("long-window protocol requires strong/weak world sizes 1/2/4")
    if config.get("warmup_steps", 0) < 200 or config.get("measured_steps", 0) < 5000:
        raise ValueError("long-window protocol requires at least 200 warm-up and 5000 steps")
    reference_keys = set(_record_index(reference))
    for number, payload in enumerate(payloads, start=1):
        if payload.get("all_executable_cases_passed") is not True:
            raise ValueError(f"repetition {number} contains a failed executable case")
        if payload.get("git_dirty") is not False:
            raise ValueError(f"repetition {number} used a dirty Git worktree")
        for key in ("git_commit", "environment", "config"):
            if payload.get(key) != reference.get(key):
                raise ValueError(f"repetition {number} differs in {key}")
        if set(_record_index(payload)) != reference_keys:
            raise ValueError(f"repetition {number} has a different case set")


def _aggregate_success(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    reference = records[0]
    for number, record in enumerate(records, start=1):
        if record.get("status") != "success":
            raise ValueError(f"repetition {number} is not successful")
        for key in IDENTITY_KEYS:
            if record.get(key) != reference.get(key):
                raise ValueError(f"repetition {number} differs in {key}")
    throughputs = [float(record["global_samples_per_second"]) for record in records]
    elapsed = [float(record["max_rank_elapsed_seconds"]) for record in records]
    memory = [int(record["peak_memory_allocated_bytes"]) for record in records]
    throughput_median = statistics.median(throughputs)
    relative_range = (max(throughputs) - min(throughputs)) / throughput_median
    return {
        **{key: reference[key] for key in IDENTITY_KEYS},
        "status": "success",
        "repeat_count": len(records),
        "throughput_median_samples_per_second": throughput_median,
        "throughput_samples_per_second": throughputs,
        "throughput_min_samples_per_second": min(throughputs),
        "throughput_max_samples_per_second": max(throughputs),
        "throughput_relative_range": relative_range,
        "stability_threshold": threshold,
        "stability_passed": relative_range <= threshold,
        "max_rank_elapsed_median_seconds": statistics.median(elapsed),
        "elapsed_samples_seconds": elapsed,
        "peak_memory_allocated_median_bytes": int(statistics.median(memory)),
        "peak_memory_samples_bytes": memory,
    }


def _aggregate_unavailable(records: list[dict[str, Any]]) -> dict[str, Any]:
    reference = records[0]
    if any(record != reference for record in records[1:]):
        raise ValueError("unavailable DDP scaling records differ across repetitions")
    return {**reference, "repeat_count": len(records)}


def aggregate_sources(
    payloads: list[dict[str, Any]],
    *,
    stability_threshold: float = 0.05,
    warning_threshold: float = 0.10,
    expected_repetitions: int = 5,
) -> dict[str, Any]:
    if not 0.0 < stability_threshold < warning_threshold < 1.0:
        raise ValueError("thresholds must satisfy 0 < stability < warning < 1")
    validate_sources(payloads, expected_repetitions)
    reference = payloads[0]
    indexes = [_record_index(payload) for payload in payloads]
    records: list[dict[str, Any]] = []
    for mode in reference["config"]["modes"]:
        for world_size in reference["config"]["world_sizes"]:
            repetitions = [index[(mode, world_size)] for index in indexes]
            statuses = {record["status"] for record in repetitions}
            if statuses == {"success"}:
                records.append(_aggregate_success(repetitions, stability_threshold))
            elif statuses == {"unavailable"}:
                records.append(_aggregate_unavailable(repetitions))
            else:
                raise ValueError(
                    f"inconsistent or unsupported statuses for {(mode, world_size)}: {statuses}"
                )
    baselines = {
        record["mode"]: record
        for record in records
        if record["status"] == "success" and record["world_size"] == 1
    }
    for record in records:
        if record["status"] != "success":
            continue
        baseline = baselines[record["mode"]]["throughput_median_samples_per_second"]
        speedup = record["throughput_median_samples_per_second"] / baseline
        record["speedup_over_world_one_median"] = speedup
        record["scaling_efficiency"] = speedup / record["world_size"]
    successful = [record for record in records if record["status"] == "success"]
    unavailable = [
        {"mode": record["mode"], "world_size": record["world_size"]}
        for record in records
        if record["status"] == "unavailable"
    ]
    unstable = [
        {"mode": record["mode"], "world_size": record["world_size"]}
        for record in successful
        if not record["stability_passed"]
    ]
    severely_unstable = [
        {"mode": record["mode"], "world_size": record["world_size"]}
        for record in successful
        if record["throughput_relative_range"] > warning_threshold
    ]
    if unavailable or severely_unstable:
        quality_status = "failed"
    elif unstable:
        quality_status = "warning"
    else:
        quality_status = "passed"
    return {
        "schema_version": 1,
        "artifact_type": "module04.ddp_scaling_long.aggregate",
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_git_commit": reference["git_commit"],
        "source_git_dirty": reference["git_dirty"],
        "environment": reference["environment"],
        "config": reference["config"],
        "measurement": {
            "aggregation": "median of five independent long-window runs",
            "repeat_count": len(payloads),
            "stability_metric": "(maximum throughput - minimum throughput) / median",
            "stability_threshold": stability_threshold,
            "warning_threshold": warning_threshold,
        },
        "status": "success",
        "correctness": {"status": "passed"},
        "measurement_quality": {
            "status": quality_status,
            "unstable_cases": unstable,
            "severely_unstable_cases": severely_unstable,
            "unavailable_cases": unavailable,
        },
        "records": records,
        "boundary": (
            "This follow-up changes only warm-up and measurement duration relative to the "
            "short-window campaign. It does not change the model, batch policy, backend, "
            "world sizes, or optimizer."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs=5, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stability-threshold", type=float, default=0.05)
    parser.add_argument("--warning-threshold", type=float, default=0.10)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.sources]
    payload = aggregate_sources(
        payloads,
        stability_threshold=args.stability_threshold,
        warning_threshold=args.warning_threshold,
        expected_repetitions=5,
    )
    payload["source_artifacts"] = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in args.sources
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(f"measurement_quality={payload['measurement_quality']['status']}")
    raise SystemExit(0 if payload["measurement_quality"]["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
