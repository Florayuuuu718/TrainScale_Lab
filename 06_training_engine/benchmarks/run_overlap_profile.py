"""Capture CUDA timelines for bucket sync, bucket async, and PyTorch DDP."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch
import torch.distributed as dist

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks"))

from artifact_contract import build_artifact, sha256_file  # noqa: E402
from torchrun_launcher import launch_torchrun  # noqa: E402
from trainscale_engine.contract import load_benchmark_config  # noqa: E402

WORKER = MODULE_ROOT / "trainscale_engine" / "worker.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bucket-cap-mb",
        type=float,
        help=(
            "Override the profiling bucket size. By default the middle value from "
            "gpu_ablation.toml is used. This option is intended for the 1 MiB overlap "
            "extension experiment."
        ),
    )
    args = parser.parse_args()
    config = load_benchmark_config(args.config)
    args.raw_directory.mkdir(parents=True, exist_ok=True)
    available = torch.cuda.device_count()
    ready = torch.cuda.is_available() and dist.is_nccl_available() and available >= 4
    records = []
    default_bucket = (
        args.bucket_cap_mb if args.bucket_cap_mb is not None else config["bucket_cap_mb"][1]
    )
    if default_bucket <= 0:
        parser.error("--bucket-cap-mb must be greater than zero")
    for strategy in ("bucket_sync", "bucket_async", "ddp"):
        if not ready:
            records.append({"strategy": strategy, "status": "unavailable"})
            continue
        job = launch_torchrun(
            repository_root=REPOSITORY_ROOT,
            worker=WORKER,
            world_size=4,
            rank_directory=args.raw_directory / strategy,
            worker_args=[
                "--mode",
                "profile",
                "--strategy",
                strategy,
                "--backend",
                "nccl",
                "--device",
                "cuda",
                "--model-preset",
                "medium",
                "--global-batch-size",
                str(config["per_rank_batch_size"] * 4),
                "--per-rank-batch-size",
                str(config["per_rank_batch_size"]),
                "--bucket-cap-bytes",
                str(round(default_bucket * 1024 * 1024)),
                "--warmup-steps",
                "5",
                "--measured-steps",
                "5",
                "--learning-rate",
                str(config["learning_rate"]),
                "--seed",
                str(config["seed"]),
                "--atol",
                "0.0001",
                "--rtol",
                "0.0001",
                "--timeout-seconds",
                str(config["timeout_seconds"]),
            ],
            python_paths=[MODULE_ROOT, REPOSITORY_ROOT / "01_pytorch_training"],
            timeout_seconds=config["timeout_seconds"],
        )
        passed = job["status"] == "success" and all(
            rank["correctness_passed"] for rank in job["ranks"]
        )
        records.append(
            {
                "strategy": strategy,
                "status": "success" if passed else "failed",
                "ranks": job["ranks"],
                "stderr_tail": job["stderr_tail"],
            }
        )
        print(f"{strategy}: {records[-1]['status']}")
    raw_artifacts = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(args.raw_directory.rglob("*"))
        if path.is_file()
    ]
    failed = any(record["status"] == "failed" for record in records)
    succeeded = any(record["status"] == "success" for record in records)
    status = "failed" if failed else "success" if succeeded else "unavailable"
    payload = build_artifact(
        artifact_type="module06.overlap_profile",
        repository_root=REPOSITORY_ROOT,
        environment={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_device_count": available,
        },
        config={
            "world_size": 4,
            "strategies": ["bucket_sync", "bucket_async", "ddp"],
            "model_preset": "medium",
            "bucket_cap_mb": default_bucket,
            "warmup_steps": 5,
            "profile_steps": 5,
            "source_config": config,
        },
        measurement={
            "profiler": "torch.profiler CPU+CUDA Chrome trace",
            "interpretation": "overlap must be confirmed from CUDA/NCCL time intervals",
        },
        status=status,
        correctness={"status": "failed" if failed else "passed" if succeeded else "not_run"},
        metrics={"records": records},
        raw_artifacts=raw_artifacts,
        boundary=(
            "Launch-before-backward-end is only a candidate; CUDA trace proves actual overlap."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
