"""Run strong/weak DDP scaling cases and record unavailable GPU world sizes honestly."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from benchmarks.launcher import launch  # noqa: E402
from trainscale_distributed.contract import (  # noqa: E402
    add_scaling_metrics,
    load_benchmark_config,
)


def environment_manifest() -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    nccl_version = torch.cuda.nccl.version() if dist.is_nccl_available() else None
    if isinstance(nccl_version, tuple):
        nccl_version = list(nccl_version)

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "nccl_available": dist.is_nccl_available(),
        "nccl_version": nccl_version,
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "gpu_compute_capabilities": [
            list(torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        ],
        "driver_versions": (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).splitlines()
            if torch.cuda.is_available()
            else []
        ),
    }


def worker_args(config: dict[str, Any], mode: str) -> list[str]:
    return [
        "--mode",
        "benchmark",
        "--backend",
        config["backend"],
        "--device",
        config["device"],
        "--seed",
        str(config["seed"]),
        "--input-dim",
        str(config["input_dim"]),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--num-classes",
        str(config["num_classes"]),
        "--global-batch-size",
        str(config["global_batch_size"]),
        "--per-rank-batch-size",
        str(config["per_rank_batch_size"]),
        "--warmup-steps",
        str(config["warmup_steps"]),
        "--measured-steps",
        str(config["measured_steps"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--scaling-mode",
        mode,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_benchmark_config(args.config)
    environment = environment_manifest()
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="trainscale-03-scaling-") as temporary:
        root = Path(temporary)
        for mode in config["modes"]:
            for world_size in config["world_sizes"]:
                if config["device"] == "cuda" and world_size > environment["cuda_device_count"]:
                    records.append(
                        {
                            "status": "unavailable",
                            "mode": mode,
                            "world_size": world_size,
                            "required_cuda_devices": world_size,
                            "available_cuda_devices": environment["cuda_device_count"],
                            "reason": "torchrun GPU jobs require one visible GPU per local process",
                        }
                    )
                    print(f"{mode}/world_size={world_size}: unavailable")
                    continue
                job = launch(
                    world_size=world_size,
                    rank_directory=root / f"{mode}_{world_size}",
                    worker_args=worker_args(config, mode),
                    timeout_seconds=args.timeout_seconds,
                )
                rank_zero = next((rank for rank in job["ranks"] if rank["rank"] == 0), None)
                if job["status"] != "success" or rank_zero is None:
                    records.append(
                        {
                            "status": "failed",
                            "mode": mode,
                            "world_size": world_size,
                            "returncode": job["returncode"],
                            "stderr_tail": job["stderr_tail"],
                        }
                    )
                    print(f"{mode}/world_size={world_size}: failed")
                    continue
                record = {key: value for key, value in rank_zero.items() if key != "rank"}
                records.append(record)
                print(
                    f"{mode}/world_size={world_size}: success "
                    f"{record['global_samples_per_second']:.2f} samples/s"
                )
    add_scaling_metrics(records)
    failures = [record for record in records if record["status"] == "failed"]
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(),
        "scope": f"{config['device']}/{config['backend']} DDP strong and weak scaling",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "environment": environment,
        "config": config,
        "records": records,
        "all_executable_cases_passed": not failures,
        "unavailable_case_count": sum(record["status"] == "unavailable" for record in records),
        "boundary": (
            "CPU ranks share one host and are a semantics/overhead experiment, "
            "not a GPU scaling proxy. "
            "Unavailable CUDA world sizes contain no fabricated throughput."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(f"all_executable_cases_passed={payload['all_executable_cases_passed']}")
    raise SystemExit(0 if payload["all_executable_cases_passed"] else 1)


if __name__ == "__main__":
    main()
