"""Run the targeted 2/4-GPU reducer ablation without a Cartesian explosion."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
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
    """Cover one-factor ablations while avoiding the full 720-job Cartesian product."""
    default_bucket = config["bucket_cap_mb"][1]
    cases: dict[tuple[Any, ...], dict[str, Any]] = {}

    def add(
        world_size: int,
        strategy: str,
        model_preset: str = "medium",
        bucket_cap_mb: float = default_bucket,
        precision: str = "fp32",
        accumulation_steps: int = 1,
    ) -> None:
        key = (
            world_size,
            strategy,
            model_preset,
            bucket_cap_mb,
            precision,
            accumulation_steps,
        )
        cases[key] = {
            "world_size": world_size,
            "strategy": strategy,
            "model_preset": model_preset,
            "bucket_cap_mb": bucket_cap_mb,
            "bucket_cap_bytes": round(bucket_cap_mb * 1024 * 1024),
            "precision": precision,
            "accumulation_steps": accumulation_steps,
        }

    for world_size in config["world_sizes"]:
        for strategy in config["strategies"]:
            add(world_size, strategy)
    for model_preset in config["model_presets"]:
        for strategy in ("bucket_async", "ddp"):
            add(4, strategy, model_preset=model_preset)
    for bucket_cap_mb in config["bucket_cap_mb"]:
        add(4, "bucket_async", bucket_cap_mb=bucket_cap_mb)
    for precision in config["precisions"]:
        for accumulation_steps in config["accumulation_steps"]:
            for strategy in ("bucket_async", "ddp"):
                add(
                    4,
                    strategy,
                    precision=precision,
                    accumulation_steps=accumulation_steps,
                )
    return list(cases.values())


def _case_id(case: dict[str, Any]) -> str:
    bucket = str(case["bucket_cap_mb"]).replace(".", "p")
    return (
        f"w{case['world_size']}-{case['strategy']}-{case['model_preset']}-"
        f"b{bucket}-{case['precision']}-a{case['accumulation_steps']}"
    )


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
            json.dumps({"config": config, "case_count": len(cases), "cases": cases}, indent=2)
            + "\n",
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
    records = []
    for case in cases:
        case_id = _case_id(case)
        if not ready:
            records.append({"case_id": case_id, **case, "status": "unavailable"})
            continue
        repetitions = []
        for repetition in range(1, config["repetitions"] + 1):
            world_size = case["world_size"]
            job = launch_torchrun(
                repository_root=REPOSITORY_ROOT,
                worker=WORKER,
                world_size=world_size,
                rank_directory=args.raw_directory / f"{case_id}-r{repetition}",
                worker_args=[
                    "--mode",
                    "benchmark",
                    "--strategy",
                    case["strategy"],
                    "--backend",
                    "nccl",
                    "--device",
                    "cuda",
                    "--model-preset",
                    case["model_preset"],
                    "--global-batch-size",
                    str(config["per_rank_batch_size"] * world_size),
                    "--per-rank-batch-size",
                    str(config["per_rank_batch_size"]),
                    "--accumulation-steps",
                    str(case["accumulation_steps"]),
                    "--bucket-cap-bytes",
                    str(case["bucket_cap_bytes"]),
                    "--precision",
                    case["precision"],
                    "--warmup-steps",
                    str(config["warmup_steps"]),
                    "--measured-steps",
                    str(config["measured_steps"]),
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
            repetitions.append(
                {
                    "status": "success" if passed else "failed",
                    "command": job["command"],
                    "rank0": job["ranks"][0] if passed else None,
                    "stderr_tail": job["stderr_tail"] if not passed else "",
                }
            )
        success = all(item["status"] == "success" for item in repetitions)
        metrics = None
        if success:
            samples = [item["rank0"] for item in repetitions]
            throughput = [sample["global_samples_per_second"] for sample in samples]
            median_throughput = statistics.median(throughput)
            metrics = {
                "median_global_samples_per_second": median_throughput,
                "throughput_relative_range": (max(throughput) - min(throughput))
                / median_throughput,
                "median_step_time_p50_ms": statistics.median(
                    sample["step_time_p50_ms"] for sample in samples
                ),
                "median_step_time_p95_ms": statistics.median(
                    sample["step_time_p95_ms"] for sample in samples
                ),
                "median_peak_memory_allocated_bytes": statistics.median(
                    sample["peak_memory_allocated_bytes"] for sample in samples
                ),
                "raw_repetitions": samples,
            }
        records.append(
            {
                "case_id": case_id,
                **case,
                "status": "success" if success else "failed",
                "metrics": metrics,
                "repetitions": repetitions,
            }
        )
        print(f"{case_id}: {records[-1]['status']}")
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
        artifact_type="module06.gpu_ablation",
        repository_root=REPOSITORY_ROOT,
        environment={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nccl": list(torch.cuda.nccl.version()) if dist.is_nccl_available() else None,
            "cuda_device_count": available,
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(available)],
        },
        config={**config, "planned_cases": cases},
        measurement={
            "design": "targeted one-factor ablations, not a full Cartesian product",
            "aggregation": f"median of {config['repetitions']} independent jobs",
            "timer": "per-step slowest-rank wall time reconstructed after measurement",
        },
        status=status,
        correctness={"status": "failed" if failed else "passed" if succeeded else "not_run"},
        metrics={"records": records},
        raw_artifacts=raw_artifacts,
        boundary="Single-node 2/4-GPU results do not establish multi-node scaling or true overlap.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
