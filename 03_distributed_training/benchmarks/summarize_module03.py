"""Validate formal module 03 results and create a content-addressed summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = (
    "environment_sm120.json",
    "gloo_semantics_cpu.json",
    "sampler_sharding_cpu.json",
    "gradient_equivalence_cpu.json",
    "checkpoint_resume_cpu.json",
    "scaling_cpu.json",
    "scaling_nccl_sm120.json",
    "ddp_profile_cpu.json",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-directory", type=Path, default=MODULE_ROOT / "results")
    parser.add_argument(
        "--output", type=Path, default=MODULE_ROOT / "results" / "module03_summary.json"
    )
    args = parser.parse_args()
    artifacts: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for name in DEFAULT_SOURCES:
        path = args.result_directory / name
        payload = load(path)
        artifacts[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scope": payload.get("scope", "environment"),
        }
        if name == "environment_sm120.json":
            gates[name] = bool(payload["distributed_available"] and payload["gloo_available"])
        elif "all_checks_passed" in payload:
            gates[name] = bool(payload["all_checks_passed"])
        else:
            gates[name] = bool(payload["all_executable_cases_passed"])
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "scope": "Module 03 local acceptance with explicit multi-GPU hardware boundary",
        "gates": gates,
        "all_executable_gates_passed": all(gates.values()),
        "source_artifacts": artifacts,
        "boundary": (
            "This machine has one GPU. 2/4/8-GPU records are unavailable, not zero-throughput "
            "measurements; CPU/Gloo results teach semantics and host contention."
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(f"all_executable_gates_passed={payload['all_executable_gates_passed']}")
    raise SystemExit(0 if payload["all_executable_gates_passed"] else 1)


if __name__ == "__main__":
    main()

