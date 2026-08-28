"""Validate and summarize the frozen Module 04 rental evidence directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_nccl.parser import correctness_passed, parse_nccl_tests_output  # noqa: E402

FORMAL_CASES = {
    "allreduce-1gpu": {"collective": "all_reduce", "devices": [0]},
    "allreduce-2gpu-near": {"collective": "all_reduce", "devices": [2, 3]},
    "allreduce-2gpu-far": {"collective": "all_reduce", "devices": [0, 2]},
    "allreduce-4gpu": {"collective": "all_reduce", "devices": [0, 1, 2, 3]},
    "allgather-2gpu-near": {"collective": "all_gather", "devices": [2, 3]},
    "broadcast-2gpu-near": {"collective": "broadcast", "devices": [2, 3]},
    "reducescatter-2gpu-near": {"collective": "reduce_scatter", "devices": [2, 3]},
}
TARGET_CASES = {
    "ddp-2gpu": {"devices": [0, 1]},
    "ddp-4gpu": {"devices": [0, 1, 2, 3]},
    "bucket-cap-2gpu": {"devices": [0, 1]},
    "bucket-cap-4gpu": {"devices": [0, 1, 2, 3]},
}
FORMAL_PATTERN = re.compile(r"^(?P<case>.+)-run(?P<run>[123])\.log$")
TARGET_PATTERN = re.compile(r"^(?P<case>.+-\d+)-run(?P<run>[123])\.log$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _median_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 3:
        raise ValueError("each NCCL row requires exactly three repetitions")
    result: dict[str, Any] = {}
    for placement in ("out_of_place", "in_place"):
        values = [row[placement] for row in rows]
        if any(value["wrong"] not in {0, None} for value in values):
            raise ValueError("cannot aggregate an NCCL row with correctness errors")
        result[placement] = {
            "median_time_us": statistics.median(value["time_us"] for value in values),
            "median_algbw_gbps": statistics.median(
                value["algbw_gbps"] for value in values
            ),
            "median_busbw_gbps": statistics.median(
                value["busbw_gbps"] for value in values
            ),
            "time_samples_us": [value["time_us"] for value in values],
            "busbw_samples_gbps": [value["busbw_gbps"] for value in values],
        }
    return result


def aggregate_nccl_directory(
    directory: Path,
    *,
    expected_cases: dict[str, dict[str, Any]],
    pattern: re.Pattern[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[tuple[int, Path, list[dict[str, Any]]]]] = defaultdict(list)
    sources: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.log")):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected NCCL log name: {path.name}")
        case = match.group("case")
        if case not in expected_cases:
            raise ValueError(f"unexpected NCCL case: {case}")
        rows = parse_nccl_tests_output(path.read_text(encoding="utf-8"))
        if not correctness_passed(rows):
            raise ValueError(f"correctness failed in {path}")
        exit_path = path.with_suffix(path.suffix + ".exit")
        if not exit_path.is_file() or exit_path.read_text(encoding="utf-8").strip() != "0":
            raise ValueError(f"missing or non-zero exit evidence for {path}")
        grouped[case].append((int(match.group("run")), path, rows))
        sources.extend(
            [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)},
                {
                    "path": str(exit_path),
                    "bytes": exit_path.stat().st_size,
                    "sha256": sha256_file(exit_path),
                },
            ]
        )
    if set(grouped) != set(expected_cases):
        raise ValueError(
            f"NCCL cases differ: actual={sorted(grouped)} expected={sorted(expected_cases)}"
        )
    aggregates: list[dict[str, Any]] = []
    for case, repetitions in sorted(grouped.items()):
        repetitions.sort(key=lambda item: item[0])
        if [item[0] for item in repetitions] != [1, 2, 3]:
            raise ValueError(f"{case} does not contain repetitions 1, 2, and 3")
        indexes = [
            {int(row["size_bytes"]): row for row in rows} for _, _, rows in repetitions
        ]
        sizes = set(indexes[0])
        if any(set(index) != sizes for index in indexes[1:]):
            raise ValueError(f"{case} repetitions contain different message sizes")
        rows = []
        for size in sorted(sizes):
            rows.append(
                {
                    "size_bytes": size,
                    **_median_row([index[size] for index in indexes]),
                }
            )
        aggregates.append(
            {
                "case": case,
                **expected_cases[case],
                "world_size": len(expected_cases[case]["devices"]),
                "repeat_count": 3,
                "rows": rows,
            }
        )
    return aggregates, sources


def aggregate_targeted(directory: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expanded: dict[str, dict[str, Any]] = {}
    for path in directory.glob("*.log"):
        match = TARGET_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected targeted NCCL log name: {path.name}")
        full_case = match.group("case")
        base_case, size = full_case.rsplit("-", maxsplit=1)
        if base_case not in TARGET_CASES:
            raise ValueError(f"unexpected targeted NCCL case: {base_case}")
        expanded[full_case] = {
            **TARGET_CASES[base_case],
            "collective": "all_reduce",
            "purpose": "gradient_payload" if base_case.startswith("ddp-") else "bucket_cap",
            "target_size_bytes": int(size),
        }
    return aggregate_nccl_directory(directory, expected_cases=expanded, pattern=TARGET_PATTERN)


def summarize_bridge(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = root / "ddp-bridge" / "ddp-bridge.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["status"] != "success" or artifact["correctness"]["status"] != "passed":
        raise ValueError("DDP bridge artifact did not pass")
    if artifact["git"]["dirty"]:
        raise ValueError("DDP bridge was collected from a dirty worktree")
    records = []
    raw_sources: list[dict[str, Any]] = []
    for raw in artifact["raw_artifacts"]:
        relative = Path(str(raw["path"]).split("/raw/", maxsplit=1)[1])
        local = root / "ddp-bridge" / "raw" / relative
        if not local.is_file() or sha256_file(local) != raw["sha256"]:
            raise ValueError(f"DDP raw artifact hash mismatch: {relative}")
        raw_sources.append(
            {"path": str(local), "bytes": local.stat().st_size, "sha256": raw["sha256"]}
        )
    for record in artifact["metrics"]["records"]:
        if record["status"] != "success":
            raise ValueError("DDP bridge contains a failed world size")
        ranks = []
        for rank in record["ranks"]:
            kernels = [
                event
                for event in rank["communication_events"]
                if str(event["name"]).startswith("ncclDevKernel_AllReduce")
            ]
            if len(kernels) != 1:
                raise ValueError("expected exactly one NCCL AllReduce kernel aggregate per rank")
            kernel = kernels[0]
            ranks.append(
                {
                    "rank": rank["rank"],
                    "bucket_sizes": rank["ddp_logging_data"]["bucket_sizes"],
                    "parameter_count": rank["parameter_count"],
                    "gradient_payload_bytes": rank["fp32_gradient_payload_bytes"],
                    "parameter_checksum_error": rank["max_parameter_checksum_error"],
                    "allreduce_count": kernel["count"],
                    "allreduce_device_time_total_us": kernel["device_time_total_us"],
                    "allreduce_device_time_per_step_us": (
                        kernel["device_time_total_us"] / kernel["count"]
                    ),
                }
            )
        records.append(
            {
                "world_size": record["world_size"],
                "rank_count": record["rank_count"],
                "communication_visible_on_all_ranks": record[
                    "communication_visible_on_all_ranks"
                ],
                "parameters_consistent_on_all_ranks": record[
                    "parameters_consistent_on_all_ranks"
                ],
                "slowest_rank_allreduce_device_time_per_step_us": max(
                    rank["allreduce_device_time_per_step_us"] for rank in ranks
                ),
                "ranks": ranks,
            }
        )
    return (
        {
            "artifact_path": str(path),
            "artifact_sha256": sha256_file(path),
            "git": artifact["git"],
            "config_sha256": artifact["config_sha256"],
            "records": records,
        },
        raw_sources,
    )


def aggregate_scaling(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = sorted((root / "ddp-scaling").glob("run[123].json"))
    if [path.name for path in paths] != ["run1.json", "run2.json", "run3.json"]:
        raise ValueError("DDP scaling requires run1.json, run2.json, and run3.json")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference = payloads[0]
    for payload in payloads:
        if not payload["all_executable_cases_passed"] or payload["git_dirty"]:
            raise ValueError("DDP scaling source failed or used a dirty worktree")
        for key in ("git_commit", "environment", "config"):
            if payload[key] != reference[key]:
                raise ValueError(f"DDP scaling repetitions differ in {key}")
    indexed = [
        {(record["mode"], record["world_size"]): record for record in payload["records"]}
        for payload in payloads
    ]
    records = []
    for mode in reference["config"]["modes"]:
        for world_size in reference["config"]["world_sizes"]:
            samples = [index[(mode, world_size)] for index in indexed]
            if {sample["status"] for sample in samples} == {"unavailable"}:
                records.append(
                    {"mode": mode, "world_size": world_size, "status": "unavailable"}
                )
                continue
            throughputs = [float(sample["global_samples_per_second"]) for sample in samples]
            throughput_median = statistics.median(throughputs)
            records.append(
                {
                    "mode": mode,
                    "world_size": world_size,
                    "status": "success",
                    "global_batch_size": samples[0]["global_batch_size"],
                    "throughput_median_samples_per_second": throughput_median,
                    "throughput_samples_per_second": throughputs,
                    "throughput_relative_range": (
                        max(throughputs) - min(throughputs)
                    )
                    / throughput_median,
                }
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
    return (
        {
            "git_commit": reference["git_commit"],
            "environment": reference["environment"],
            "config": reference["config"],
            "repeat_count": 3,
            "records": records,
            "limitation": (
                "World-size-one throughput is unstable when relative_range is large; "
                "speedup remains a median-based descriptive result, not a precise estimate."
            ),
        },
        [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_root.resolve()
    formal, formal_sources = aggregate_nccl_directory(
        root / "formal", expected_cases=FORMAL_CASES, pattern=FORMAL_PATTERN
    )
    targeted, targeted_sources = aggregate_targeted(root / "targeted-ddp-payload")
    bridge, bridge_sources = summarize_bridge(root)
    scaling, scaling_sources = aggregate_scaling(root)
    identity = (root / "project-git-commit.txt").read_text(encoding="utf-8").strip()
    if identity != bridge["git"]["commit"] or identity != scaling["git_commit"]:
        raise ValueError("project Git identity differs across evidence")
    topology_paths = sorted((root / "topology-probe").glob("*"))
    topology_sources = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in topology_paths
        if path.is_file()
    ]
    payload = {
        "schema_version": 1,
        "artifact_type": "module04.rental_analysis",
        "status": "success",
        "correctness": {"status": "passed"},
        "source_git_commit": identity,
        "source_git_dirty": False,
        "nccl_formal": formal,
        "nccl_targeted": targeted,
        "ddp_bridge": bridge,
        "ddp_scaling": scaling,
        "topology": {
            "gpu_pairs": {
                "near": {"devices": [2, 3], "reported_path": "PHB"},
                "far": {"devices": [0, 2], "reported_path": "SYS"},
            },
            "gpu_p2p_capability": "CNS (chipset not supported) for every pair",
            "observed_nccl_transport": "SHM/direct/direct for both near and far probes",
            "interpretation": (
                "The topology labels differ, but both measured pairs used the same SHM "
                "transport on this host. A small near/far difference must not be generalized "
                "to hosts with working GPU P2P or NVLink."
            ),
        },
        "source_artifacts": sorted(
            [
                *formal_sources,
                *targeted_sources,
                *bridge_sources,
                *scaling_sources,
                *topology_sources,
            ],
            key=lambda item: item["path"],
        ),
        "boundary": (
            "Single-host 4x RTX 4090 D, no NVLink, GPU P2P unsupported, NCCL used SHM. "
            "Results explain this frozen teaching workload and are not a hardware ranking."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print("status=success correctness=passed")


if __name__ == "__main__":
    main()
