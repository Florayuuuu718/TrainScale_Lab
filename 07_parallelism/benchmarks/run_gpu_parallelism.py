"""Run the targeted Module 07 correctness gates and 2/4-GPU benchmarks."""

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
from trainscale_parallel.contract import load_gpu_config  # noqa: E402

WORKER = MODULE_ROOT / "trainscale_parallel" / "worker.py"


def planned_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Cover strategy, world size, model size, and wrap granularity without a grid."""
    cases: dict[tuple[Any, ...], dict[str, Any]] = {}

    def add(world_size: int, strategy: str, model_preset: str) -> None:
        cases[(world_size, strategy, model_preset)] = {
            "world_size": world_size,
            "strategy": strategy,
            "model_preset": model_preset,
        }

    for world_size in config["world_sizes"]:
        for strategy in ("ddp", "fsdp_root", "fsdp_layer"):
            add(world_size, strategy, "medium")
    for strategy in ("ddp", "fsdp_layer"):
        add(4, strategy, "small")
    add(1, "tp_reference", "medium")
    for world_size in config["world_sizes"]:
        add(world_size, "tp", "medium")
    return list(cases.values())


def planned_preflights(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"world_size": world_size, "mode": mode}
        for world_size in config["world_sizes"]
        for mode in ("fsdp2_probe", "native_tp_probe")
    ]


def _case_id(case: dict[str, Any]) -> str:
    return f"w{case['world_size']}-{case['strategy']}-{case['model_preset']}"


def _launch(
    *,
    world_size: int,
    rank_directory: Path,
    worker_args: list[str],
    timeout_seconds: int,
) -> dict[str, Any]:
    return launch_torchrun(
        repository_root=REPOSITORY_ROOT,
        worker=WORKER,
        world_size=world_size,
        rank_directory=rank_directory,
        worker_args=worker_args,
        python_paths=[MODULE_ROOT, REPOSITORY_ROOT / "06_training_engine"],
        timeout_seconds=timeout_seconds,
    )


def _run_preflight(
    case: dict[str, Any], config: dict[str, Any], raw_directory: Path
) -> dict[str, Any]:
    world_size = case["world_size"]
    mode = case["mode"]
    case_id = f"preflight-w{world_size}-{mode}"
    job = _launch(
        world_size=world_size,
        rank_directory=raw_directory / case_id,
        worker_args=[
            "--mode",
            mode,
            "--backend",
            "nccl",
            "--device",
            "cuda",
            "--batch-size",
            "4",
            "--learning-rate",
            str(config["learning_rate"]),
            "--seed",
            str(config["seed"]),
            "--atol",
            "0.0001",
            "--timeout-seconds",
            str(config["timeout_seconds"]),
        ],
        timeout_seconds=config["timeout_seconds"],
    )
    passed = job["status"] == "success" and all(rank["correctness_passed"] for rank in job["ranks"])
    return {
        "case_id": case_id,
        **case,
        "status": "success" if passed else "failed",
        "ranks": job["ranks"],
        "command": job["command"],
        "stderr_tail": "" if passed else job["stderr_tail"],
    }


def _run_benchmark_repetition(
    case: dict[str, Any],
    config: dict[str, Any],
    rank_directory: Path,
    repetition: int,
) -> dict[str, Any]:
    job = _launch(
        world_size=case["world_size"],
        rank_directory=rank_directory,
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
            "--per-rank-batch-size",
            str(config["per_rank_batch_size"]),
            "--warmup-steps",
            str(config["warmup_steps"]),
            "--measured-steps",
            str(config["measured_steps"]),
            "--learning-rate",
            str(config["learning_rate"]),
            "--seed",
            str(config["seed"] + repetition),
            "--timeout-seconds",
            str(config["timeout_seconds"]),
        ],
        timeout_seconds=config["timeout_seconds"],
    )
    success = job["status"] == "success"
    rank0 = job["ranks"][0] if success else None
    metrics: dict[str, Any] | None = None
    if success:
        assert isinstance(rank0, dict)
        metrics = {
            **rank0,
            "maximum_peak_memory_allocated_bytes": max(
                rank["peak_memory_allocated_bytes"] for rank in job["ranks"]
            ),
            "maximum_local_parameter_bytes": max(
                rank["local_parameter_bytes"] for rank in job["ranks"]
            ),
        }
    return {
        "status": "success" if success else "failed",
        "command": job["command"],
        "metrics": metrics,
        "stderr_tail": "" if success else job["stderr_tail"],
    }


def _aggregate(repetitions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not all(item["status"] == "success" for item in repetitions):
        return None
    samples = [item["metrics"] for item in repetitions]
    throughput = [sample["samples_per_second"] for sample in samples]
    median_throughput = statistics.median(throughput)
    return {
        "median_samples_per_second": median_throughput,
        "throughput_relative_range": (max(throughput) - min(throughput)) / median_throughput,
        "median_step_time_p50_ms": statistics.median(
            sample["step_time_p50_ms"] for sample in samples
        ),
        "median_step_time_p95_ms": statistics.median(
            sample["step_time_p95_ms"] for sample in samples
        ),
        "median_maximum_peak_memory_allocated_bytes": statistics.median(
            sample["maximum_peak_memory_allocated_bytes"] for sample in samples
        ),
        "median_maximum_local_parameter_bytes": statistics.median(
            sample["maximum_local_parameter_bytes"] for sample in samples
        ),
        "raw_repetitions": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_gpu_config(args.config)
    cases = planned_cases(config)
    preflights = planned_preflights(config)
    if args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "config": config,
                    "preflight_count": len(preflights),
                    "preflights": preflights,
                    "benchmark_case_count": len(cases),
                    "benchmark_job_count": len(cases) * config["repetitions"],
                    "cases": cases,
                },
                indent=2,
            )
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
    preflight_records: list[dict[str, Any]] = []
    benchmark_records: list[dict[str, Any]] = []
    if ready:
        for case in preflights:
            record = _run_preflight(case, config, args.raw_directory)
            preflight_records.append(record)
            print(f"{record['case_id']}: {record['status']}")
        gates_passed = all(record["status"] == "success" for record in preflight_records)
        if gates_passed:
            for case in cases:
                case_id = _case_id(case)
                repetitions = [
                    _run_benchmark_repetition(
                        case,
                        config,
                        args.raw_directory / f"{case_id}-r{repetition}",
                        repetition,
                    )
                    for repetition in range(1, config["repetitions"] + 1)
                ]
                aggregate = _aggregate(repetitions)
                benchmark_records.append(
                    {
                        "case_id": case_id,
                        **case,
                        "status": "success" if aggregate is not None else "failed",
                        "metrics": aggregate,
                        "repetitions": repetitions,
                    }
                )
                print(f"{case_id}: {benchmark_records[-1]['status']}")
    else:
        preflight_records = [{**case, "status": "unavailable"} for case in preflights]
        benchmark_records = [
            {"case_id": _case_id(case), **case, "status": "unavailable"} for case in cases
        ]
    failed = any(record["status"] == "failed" for record in preflight_records + benchmark_records)
    succeeded = bool(benchmark_records) and all(
        record["status"] == "success" for record in benchmark_records
    )
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
        artifact_type="module07.gpu_parallelism",
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
        config={**config, "preflights": preflights, "planned_cases": cases},
        measurement={
            "design": "correctness-gated targeted comparisons, not a Cartesian grid",
            "aggregation": f"median of {config['repetitions']} independent jobs",
            "timer": "slowest-rank step time; CUDA synchronized",
            "batch_semantics": (
                "DDP/FSDP global batch scales with world size; TP batch is replicated"
            ),
        },
        status=status,
        correctness={
            "status": "failed" if failed else "passed" if succeeded else "not_run",
            "preflights": preflight_records,
        },
        metrics={"records": benchmark_records},
        raw_artifacts=raw_artifacts,
        boundary=(
            "Single-node results do not establish multi-node scaling. TP benchmarks only the "
            "Transformer MLP core; 2D parallelism remains outside the default gate."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
