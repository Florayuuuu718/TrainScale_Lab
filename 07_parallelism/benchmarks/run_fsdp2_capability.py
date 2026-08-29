"""Probe whether the current CPU/Gloo environment can execute real FSDP2 semantics."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks"))

from artifact_contract import build_artifact, sha256_file  # noqa: E402
from torchrun_launcher import launch_torchrun  # noqa: E402

WORKER = MODULE_ROOT / "trainscale_parallel" / "worker.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.raw_directory.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "world_size": 2,
        "backend": "gloo",
        "device": "cpu",
        "model_preset": "small",
        "batch_size": 4,
        "learning_rate": 0.05,
        "seed": 20260824,
        "atol": 1e-5,
        "timeout_seconds": 120,
    }
    job = launch_torchrun(
        repository_root=REPOSITORY_ROOT,
        worker=WORKER,
        world_size=config["world_size"],
        rank_directory=args.raw_directory / "fsdp2-cpu-probe",
        worker_args=[
            "--mode",
            "fsdp2_probe",
            "--backend",
            "gloo",
            "--device",
            "cpu",
            "--batch-size",
            str(config["batch_size"]),
            "--learning-rate",
            str(config["learning_rate"]),
            "--seed",
            str(config["seed"]),
            "--atol",
            str(config["atol"]),
            "--timeout-seconds",
            str(config["timeout_seconds"]),
        ],
        python_paths=[MODULE_ROOT, REPOSITORY_ROOT / "06_training_engine"],
        timeout_seconds=config["timeout_seconds"],
    )
    success = job["status"] == "success" and all(
        rank["correctness_passed"] for rank in job["ranks"]
    )
    raw_artifacts = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(args.raw_directory.rglob("*"))
        if path.is_file()
    ]
    status = "success" if success else "unavailable"
    payload = build_artifact(
        artifact_type="module07.fsdp2_cpu_capability",
        repository_root=REPOSITORY_ROOT,
        environment={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "backend": "gloo",
            "device": "cpu",
        },
        config=config,
        measurement={"probe": "real torch.distributed.fsdp.fully_shard one-step update"},
        status=status,
        correctness={"status": "passed" if success else "not_run"},
        metrics={
            "ranks": job["ranks"],
            "returncode": job["returncode"],
            "timed_out": job["timed_out"],
            "stderr_tail": job["stderr_tail"],
        },
        raw_artifacts=raw_artifacts,
        boundary=(
            "CPU support is an environment capability probe; CUDA FSDP2 memory and performance "
            "still require the multi-GPU gate."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
