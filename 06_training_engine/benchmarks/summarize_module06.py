"""Combine Module 06 local and optional GPU artifacts into an acceptance record."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(
    baseline: dict[str, Any],
    correctness: dict[str, Any],
    gpu_ablation: dict[str, Any] | None,
    overlap_profile: dict[str, Any] | None,
    amp_overflow: dict[str, Any] | None,
) -> dict[str, Any]:
    local_ready = baseline.get("status") == "success" and correctness.get("status") == "success"
    gpu_status = "not_provided" if gpu_ablation is None else gpu_ablation.get("status")
    profile_status = "not_provided" if overlap_profile is None else overlap_profile.get("status")
    overflow_status = "not_provided" if amp_overflow is None else amp_overflow.get("status")
    complete = (
        local_ready
        and gpu_status == "success"
        and profile_status == "success"
        and overflow_status == "success"
    )
    return {
        "schema_version": 1,
        "recorded_at": datetime.now().astimezone().isoformat(),
        "status": "complete" if complete else "passed_local_gates" if local_ready else "failed",
        "gates": {
            "single_device_baseline": baseline.get("status"),
            "cpu_gloo_correctness": correctness.get("status"),
            "gpu_ablation": gpu_status,
            "overlap_profile": profile_status,
            "amp_overflow": overflow_status,
        },
        "local_evidence": {
            "baseline_loss_ratio": baseline.get("metrics", {}).get("loss_ratio"),
            "correctness_case_count": len(correctness.get("metrics", {}).get("records", [])),
            "maximum_gradient_error": max(
                (
                    rank.get("gradient_max_error", 0.0)
                    for record in correctness.get("metrics", {}).get("records", [])
                    for rank in record.get("ranks", [])
                ),
                default=None,
            ),
            "maximum_parameter_error": max(
                (
                    rank.get("parameter_max_error", 0.0)
                    for record in correctness.get("metrics", {}).get("records", [])
                    for rank in record.get("ranks", [])
                ),
                default=None,
            ),
        },
        "pending_gpu_gates": []
        if complete
        else [
            "2/4-GPU targeted reducer ablation",
            "4-GPU bucket-sync/bucket-async/DDP CUDA timeline",
            "AMP overflow/skip and accumulation=4 evidence",
        ],
        "boundary": (
            "GPU ablation, profiler, and overflow gates passed on the recorded hardware; "
            "their throughput and overlap conclusions do not automatically transfer to other "
            "models, bucket plans, interconnects, or software versions."
            if complete
            else "Local gates prove trainability, global-batch equivalence, accumulation, "
            "None-gradient, checkpoint, and reducer lifecycle. They do not prove CUDA/NCCL "
            "overlap or speedup."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--gpu-ablation", type=Path)
    parser.add_argument("--overlap-profile", type=Path)
    parser.add_argument("--amp-overflow", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(
        _read(args.baseline),
        _read(args.correctness),
        _read(args.gpu_ablation) if args.gpu_ablation else None,
        _read(args.overlap_profile) if args.overlap_profile else None,
        _read(args.amp_overflow) if args.amp_overflow else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(0 if payload["status"] != "failed" else 1)


if __name__ == "__main__":
    main()
