"""Verify AMP overflow detection and optimizer-step skipping on 2/4 GPUs."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

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


def planned_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"world_size": world_size, "strategy": strategy}
        for world_size in config["world_sizes"]
        for strategy in ("bucket_async", "ddp")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_benchmark_config(args.config)
    cases = planned_cases(config)
    if args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"case_count": len(cases), "cases": cases}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(args.output)
        return
    if args.raw_directory is None:
        raise ValueError("--raw-directory is required unless --dry-run is used")
    args.raw_directory.mkdir(parents=True, exist_ok=True)
    available = torch.cuda.device_count()
    ready = (
        torch.cuda.is_available()
        and dist.is_nccl_available()
        and available >= max(config["world_sizes"])
    )
    records: list[dict[str, Any]] = []
    for case in cases:
        case_id = f"w{case['world_size']}-{case['strategy']}"
        if not ready:
            records.append({"case_id": case_id, **case, "status": "unavailable"})
            continue
        job = launch_torchrun(
            repository_root=REPOSITORY_ROOT,
            worker=WORKER,
            world_size=case["world_size"],
            rank_directory=args.raw_directory / case_id,
            worker_args=[
                "--mode",
                "amp_overflow_probe",
                "--strategy",
                case["strategy"],
                "--backend",
                "nccl",
                "--device",
                "cuda",
                "--model-preset",
                "small",
                "--per-rank-batch-size",
                "8",
                "--bucket-cap-bytes",
                str(round(config["bucket_cap_mb"][1] * 1024 * 1024)),
                "--learning-rate",
                str(config["learning_rate"]),
                "--seed",
                str(config["seed"]),
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
                "case_id": case_id,
                **case,
                "status": "success" if passed else "failed",
                "ranks": job["ranks"],
                "command": job["command"],
                "stderr_tail": "" if passed else job["stderr_tail"],
            }
        )
        print(f"{case_id}: {records[-1]['status']}")
    failed = any(record["status"] == "failed" for record in records)
    succeeded = records and all(record["status"] == "success" for record in records)
    status = "failed" if failed else "success" if succeeded else "unavailable"
    raw_artifacts = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(args.raw_directory.rglob("*"))
        if path.is_file()
    ]
    payload = build_artifact(
        artifact_type="module06.amp_overflow_probe",
        repository_root=REPOSITORY_ROOT,
        environment={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_device_count": available,
        },
        config={
            "source_config": config,
            "cases": cases,
            "model_preset": "small",
            "injection_point": "after reducer completion and before GradScaler.unscale_",
        },
        measurement={
            "checks": [
                "finite AMP step updates parameters",
                "injected non-finite gradient lowers scale",
                "overflow skips optimizer step",
                "parameters remain bitwise unchanged on skipped step",
            ]
        },
        status=status,
        correctness={"status": "failed" if failed else "passed" if succeeded else "not_run"},
        metrics={"records": records},
        raw_artifacts=raw_artifacts,
        boundary="Synthetic overflow validates GradScaler control flow, not natural overflow rate.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
