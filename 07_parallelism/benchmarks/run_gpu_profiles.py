"""Capture 4-GPU DDP, FSDP2, and TP traces after correctness gates pass."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
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
CASES = (
    {"strategy": "ddp", "expected_collective": "allreduce"},
    {"strategy": "fsdp_layer", "expected_collective": "allgather/reducescatter"},
    {"strategy": "tp", "expected_collective": "allreduce"},
)


def _collective_names(trace: Path) -> dict[str, int]:
    payload = json.loads(trace.read_text(encoding="utf-8"))
    names = Counter(
        str(event.get("name", ""))
        for event in payload.get("traceEvents", [])
        if any(
            token in str(event.get("name", "")).lower().replace("_", "")
            for token in ("allreduce", "allgather", "reducescatter")
        )
    )
    return dict(sorted(names.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--correctness-artifact", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_gpu_config(args.config)
    correctness = json.loads(args.correctness_artifact.read_text(encoding="utf-8"))
    if (
        correctness.get("status") != "success"
        or correctness.get("correctness", {}).get("status") != "passed"
    ):
        raise ValueError("GPU correctness artifact must pass before profiling")
    args.raw_directory.mkdir(parents=True, exist_ok=True)
    available = torch.cuda.device_count()
    ready = torch.cuda.is_available() and dist.is_nccl_available() and available >= 4
    records: list[dict[str, Any]] = []
    for case in CASES:
        case_id = f"w4-{case['strategy']}-profile"
        if not ready:
            records.append({"case_id": case_id, **case, "status": "unavailable"})
            continue
        rank_directory = args.raw_directory / case_id
        trace_directory = rank_directory / "traces"
        job = launch_torchrun(
            repository_root=REPOSITORY_ROOT,
            worker=WORKER,
            world_size=4,
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
                "medium",
                "--per-rank-batch-size",
                str(config["per_rank_batch_size"]),
                "--warmup-steps",
                "5",
                "--measured-steps",
                "5",
                "--learning-rate",
                str(config["learning_rate"]),
                "--seed",
                str(config["seed"]),
                "--trace-directory",
                str(trace_directory),
                "--timeout-seconds",
                str(config["timeout_seconds"]),
            ],
            python_paths=[MODULE_ROOT, REPOSITORY_ROOT / "06_training_engine"],
            timeout_seconds=config["timeout_seconds"],
        )
        traces = sorted(trace_directory.glob("rank_*.json"))
        collective_names = {trace.name: _collective_names(trace) for trace in traces}
        success = job["status"] == "success" and len(traces) == 4
        records.append(
            {
                "case_id": case_id,
                **case,
                "status": "success" if success else "failed",
                "trace_count": len(traces),
                "collective_event_names": collective_names,
                "command": job["command"],
                "stderr_tail": "" if success else job["stderr_tail"],
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
        artifact_type="module07.gpu_profiles",
        repository_root=REPOSITORY_ROOT,
        environment={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_device_count": available,
        },
        config={"world_size": 4, "cases": list(CASES), "measured_steps": 5},
        measurement={
            "profiler": "torch.profiler CPU+CUDA Chrome trace",
            "purpose": "identify strategy-specific collective events, not performance",
        },
        status=status,
        correctness={
            "status": "passed" if succeeded else "failed" if failed else "not_run",
            "source_artifact": str(args.correctness_artifact.resolve()),
            "source_sha256": sha256_file(args.correctness_artifact),
        },
        metrics={"records": records},
        raw_artifacts=raw_artifacts,
        boundary="Profiler timings are perturbed and must not replace formal benchmark timings.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
