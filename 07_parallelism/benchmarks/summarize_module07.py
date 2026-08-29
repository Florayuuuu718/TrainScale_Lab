"""Combine Module 07 local gates and optional GPU artifacts into acceptance status."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(
    memory: dict[str, Any],
    tp_correctness: dict[str, Any],
    fsdp2_capability: dict[str, Any],
    native_tp_capability: dict[str, Any],
    gpu_parallelism: dict[str, Any] | None,
    gpu_profiles: dict[str, Any] | None,
) -> dict[str, Any]:
    local = {
        "memory_model": memory.get("status"),
        "custom_tp_cpu_gloo": tp_correctness.get("status"),
        "fsdp2_cpu_gloo": fsdp2_capability.get("status"),
        "native_tp_cpu_gloo": native_tp_capability.get("status"),
    }
    local_ready = all(status == "success" for status in local.values())
    gpu_status = "not_provided" if gpu_parallelism is None else gpu_parallelism.get("status")
    profile_status = "not_provided" if gpu_profiles is None else gpu_profiles.get("status")
    complete = local_ready and gpu_status == "success" and profile_status == "success"
    tp_errors = [
        rank.get("maximum_error", 0.0)
        for record in tp_correctness.get("metrics", {}).get("records", [])
        for rank in record.get("ranks", [])
    ]
    fsdp_errors = [
        rank.get("maximum_error", 0.0)
        for rank in fsdp2_capability.get("metrics", {}).get("ranks", [])
    ]
    native_errors = [
        rank.get("maximum_error", 0.0)
        for rank in native_tp_capability.get("metrics", {}).get("ranks", [])
    ]
    return {
        "schema_version": 1,
        "recorded_at": datetime.now().astimezone().isoformat(),
        "status": "complete" if complete else "passed_local_gates" if local_ready else "failed",
        "gates": {
            **local,
            "gpu_parallelism": gpu_status,
            "gpu_profiles": profile_status,
        },
        "local_evidence": {
            "memory_model_record_count": len(memory.get("metrics", {}).get("records", [])),
            "tp_case_count": len(tp_correctness.get("metrics", {}).get("records", [])),
            "maximum_custom_tp_error": max(tp_errors, default=None),
            "maximum_fsdp2_error": max(fsdp_errors, default=None),
            "maximum_native_tp_error": max(native_errors, default=None),
        },
        "pending_gpu_gates": []
        if complete
        else [
            "2/4-GPU FSDP2 and native TP correctness preflights",
            "DDP/FSDP2/TP throughput and peak-memory comparison",
            "4-GPU collective traces for DDP, FSDP2, and TP",
            "controlled single-GPU OOM transition if the configured model can expose one",
        ],
        "optional_not_blocking": [
            "2D TP x DP/FSDP",
            "pipeline parallelism",
            "multi-node scaling",
        ],
        "boundary": (
            "Local gates validate sharding math, DTensor placements, FSDP2 one-step updates, and "
            "distributed checkpoint resume. They do not prove CUDA memory savings or speedup."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--tp-correctness", type=Path, required=True)
    parser.add_argument("--fsdp2-capability", type=Path, required=True)
    parser.add_argument("--native-tp-capability", type=Path, required=True)
    parser.add_argument("--gpu-parallelism", type=Path)
    parser.add_argument("--gpu-profiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(
        _read(args.memory),
        _read(args.tp_correctness),
        _read(args.fsdp2_capability),
        _read(args.native_tp_capability),
        _read(args.gpu_parallelism) if args.gpu_parallelism else None,
        _read(args.gpu_profiles) if args.gpu_profiles else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(0 if payload["status"] != "failed" else 1)


if __name__ == "__main__":
    main()
