"""Validate repeated scaling runs and aggregate them with a median estimator."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_distributed.contract import add_scaling_metrics  # noqa: E402

FORMAL_RUN_NAMES = (
    "gpu_formal_run1.json",
    "gpu_formal_run2.json",
    "gpu_formal_run3.json",
)
EVIDENCE_NAMES = (
    "environment.json",
    "gpu-list.txt",
    "gpu-topology.txt",
    "nvidia-smi.txt",
    "gpu_smoke.json",
    *FORMAL_RUN_NAMES,
)
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def topology_summary(text: str) -> dict[str, Any]:
    gpu_rows = [line for line in text.splitlines() if re.match(r"^GPU\d+\s", line)]
    numa_affinities = sorted(
        {
            fields[-2]
            for line in gpu_rows
            if len(fields := line.split()) >= 3 and fields[-2].isdigit()
        }
    )
    return {
        "gpu_row_count": len(gpu_rows),
        "contains_nvlink": any(re.search(r"\bNV\d+\b", line) for line in gpu_rows),
        "contains_cross_numa_sys_path": any("SYS" in line.split() for line in gpu_rows),
        "numa_affinities": numa_affinities,
        "interpretation": (
            "GPU0/1 and GPU2/3 form two NUMA-local pairs; cross-pair paths traverse SYS. "
            "No NVLink path was reported."
        ),
    }


def _record_index(payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("each run must contain a records list")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (str(record["mode"]), int(record["world_size"]))
        if key in indexed:
            raise ValueError(f"duplicate record {key}")
        indexed[key] = record
    return indexed


def _validate_runs(payloads: list[dict[str, Any]]) -> None:
    if len(payloads) != 3:
        raise ValueError("formal cloud aggregation requires exactly three repetitions")
    reference = payloads[0]
    reference_keys = set(_record_index(reference))
    for index, payload in enumerate(payloads, start=1):
        if payload.get("all_executable_cases_passed") is not True:
            raise ValueError(f"run {index} contains a failed executable case")
        if payload.get("git_dirty") is not False:
            raise ValueError(f"run {index} did not use a clean Git worktree")
        for key in ("git_commit", "environment", "config"):
            if payload.get(key) != reference.get(key):
                raise ValueError(f"run {index} differs in {key}")
        if set(_record_index(payload)) != reference_keys:
            raise ValueError(f"run {index} has a different case set")


def _aggregate_success(records: list[dict[str, Any]]) -> dict[str, Any]:
    reference = records[0]
    for index, record in enumerate(records, start=1):
        if record.get("status") != "success":
            raise ValueError(f"repetition {index} is not successful")
        for key in IDENTITY_KEYS:
            if record.get(key) != reference.get(key):
                raise ValueError(f"repetition {index} differs in {key}")
    throughputs = [float(record["global_samples_per_second"]) for record in records]
    elapsed = [float(record["max_rank_elapsed_seconds"]) for record in records]
    memory = [int(record["peak_memory_allocated_bytes"]) for record in records]
    throughput_median = median(throughputs)
    return {
        **{key: reference[key] for key in IDENTITY_KEYS},
        "status": "success",
        "worker_mode": reference.get("worker_mode", "benchmark"),
        "repeat_count": len(records),
        "global_samples_per_second": throughput_median,
        "throughput_samples_per_second": throughputs,
        "throughput_min_samples_per_second": min(throughputs),
        "throughput_max_samples_per_second": max(throughputs),
        "throughput_relative_range": (
            (max(throughputs) - min(throughputs)) / throughput_median
        ),
        "max_rank_elapsed_seconds": median(elapsed),
        "elapsed_samples_seconds": elapsed,
        "peak_memory_allocated_bytes": int(median(memory)),
        "peak_memory_samples_bytes": memory,
    }


def _aggregate_unavailable(records: list[dict[str, Any]]) -> dict[str, Any]:
    reference = records[0]
    for index, record in enumerate(records, start=1):
        if record != reference:
            raise ValueError(f"unavailable repetition {index} differs from the first run")
    return {**reference, "repeat_count": len(records)}


def aggregate(evidence_directory: Path, archive_sha256: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", archive_sha256):
        raise ValueError("archive SHA-256 must contain exactly 64 hexadecimal characters")
    paths = {name: evidence_directory / name for name in EVIDENCE_NAMES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"missing evidence files: {missing}")

    payloads = [load_json(paths[name]) for name in FORMAL_RUN_NAMES]
    _validate_runs(payloads)
    reference = payloads[0]
    indexes = [_record_index(payload) for payload in payloads]
    config = reference["config"]
    records: list[dict[str, Any]] = []
    for mode in config["modes"]:
        for world_size in config["world_sizes"]:
            repetitions = [index[(mode, world_size)] for index in indexes]
            statuses = {record["status"] for record in repetitions}
            if len(statuses) != 1:
                raise ValueError(f"inconsistent statuses for {(mode, world_size)}")
            if statuses == {"success"}:
                records.append(_aggregate_success(repetitions))
            elif statuses == {"unavailable"}:
                records.append(_aggregate_unavailable(repetitions))
            else:
                raise ValueError(f"unsupported status for {(mode, world_size)}: {statuses}")
    add_scaling_metrics(records)

    environment = load_json(paths["environment.json"])
    for key in ("platform", "python", "torch", "torch_cuda_runtime", "cuda_device_count"):
        if environment.get(key) != reference["environment"].get(key):
            raise ValueError(f"environment evidence differs from formal runs in {key}")
    success_count = sum(record["status"] == "success" for record in records)
    unavailable_count = sum(record["status"] == "unavailable" for record in records)
    return {
        "schema_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(),
        "scope": "AutoDL single-node 4x RTX 4090 D NCCL strong and weak scaling",
        "aggregation": {
            "method": "median of three independent runs per mode/world-size case",
            "speedup_note": "speedup and efficiency are recomputed from median throughput",
            "source_run_count": len(payloads),
        },
        "source_git_commit": reference["git_commit"],
        "source_git_dirty": reference["git_dirty"],
        "downloaded_archive_sha256": archive_sha256.lower(),
        "source_artifacts": {
            name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in paths.items()
        },
        "environment": environment,
        "topology": topology_summary(paths["gpu-topology.txt"].read_text(encoding="utf-8")),
        "config": config,
        "records": records,
        "successful_case_count": success_count,
        "unavailable_case_count": unavailable_count,
        "all_executable_cases_passed": success_count == 6 and unavailable_count == 2,
        "boundary": (
            "This is a short synthetic-model, single-node scaling experiment. It proves real "
            "1/2/4-GPU NCCL execution, not production-model linear scaling. Eight-GPU cases "
            "remain unavailable because the rented instance exposed four GPUs."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = aggregate(args.evidence_directory, args.archive_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(f"all_executable_cases_passed={payload['all_executable_cases_passed']}")
    raise SystemExit(0 if payload["all_executable_cases_passed"] else 1)


if __name__ == "__main__":
    main()
