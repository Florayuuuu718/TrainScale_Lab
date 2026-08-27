"""Launch the module 03 workload under a multi-GPU DDP/NCCL profiler."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
WORKER = MODULE_ROOT / "trainscale_nccl" / "ddp_bridge_worker.py"
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks"))

from artifact_contract import build_artifact, sha256_file  # noqa: E402
from trainscale_nccl.contract import load_bridge_config  # noqa: E402
from trainscale_nccl.environment import collect_environment  # noqa: E402


def torchrun_command(
    *,
    world_size: int,
    config: dict[str, Any],
    rank_directory: Path,
    trace_directory: Path,
    timeout_seconds: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
        str(WORKER),
        "--rank-directory",
        str(rank_directory),
        "--trace-directory",
        str(trace_directory),
        "--seed",
        str(config["seed"]),
        "--input-dim",
        str(config["input_dim"]),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--num-classes",
        str(config["num_classes"]),
        "--per-rank-batch-size",
        str(config["per_rank_batch_size"]),
        "--warmup-steps",
        str(config["warmup_steps"]),
        "--profile-steps",
        str(config["profile_steps"]),
        "--bucket-cap-mb",
        str(config["bucket_cap_mb"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--timeout-seconds",
        str(timeout_seconds),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    config = load_bridge_config(args.config)
    environment = collect_environment()
    records: list[dict[str, Any]] = []
    raw_artifacts: list[dict[str, Any]] = []
    for world_size in config["world_sizes"]:
        if not environment["module04_capability"]["multi_gpu_ready"] or (
            environment["cuda_device_count"] < world_size
        ):
            records.append(
                {
                    "world_size": world_size,
                    "status": "unavailable",
                    "reason": (
                        f"DDP bridge requires Linux/NCCL and {world_size} visible GPUs; "
                        f"available={environment['cuda_device_count']}"
                    ),
                }
            )
            print(f"world_size={world_size}: unavailable")
            continue
        rank_directory = args.raw_directory / f"world_{world_size}" / "ranks"
        trace_directory = args.raw_directory / f"world_{world_size}" / "traces"
        command = torchrun_command(
            world_size=world_size,
            config=config,
            rank_directory=rank_directory,
            trace_directory=trace_directory,
            timeout_seconds=args.timeout_seconds,
        )
        process_environment = os.environ.copy()
        process_environment["PYTHONPATH"] = str(MODULE_ROOT)
        try:
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                env=process_environment,
                capture_output=True,
                check=False,
                text=True,
                timeout=args.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            records.append(
                {
                    "world_size": world_size,
                    "status": "failed",
                    "reason": f"torchrun timed out after {args.timeout_seconds} seconds",
                    "stderr_tail": (error.stderr or "")[-4000:],
                }
            )
            continue
        rank_files = sorted(rank_directory.glob("rank_*.json"))
        ranks = [json.loads(path.read_text(encoding="utf-8")) for path in rank_files]
        communication_visible = all(rank["communication_events"] for rank in ranks)
        parameters_consistent = all(rank["parameters_consistent"] for rank in ranks)
        success = (
            completed.returncode == 0
            and len(ranks) == world_size
            and communication_visible
            and parameters_consistent
        )
        records.append(
            {
                "world_size": world_size,
                "status": "success" if success else "failed",
                "command": command,
                "returncode": completed.returncode,
                "rank_count": len(ranks),
                "communication_visible_on_all_ranks": communication_visible,
                "parameters_consistent_on_all_ranks": parameters_consistent,
                "ranks": ranks,
                "stderr_tail": completed.stderr[-4000:],
            }
        )
        for path in [*rank_files, *sorted(trace_directory.glob("rank_*_trace.json"))]:
            raw_artifacts.append({"path": str(path), "sha256": sha256_file(path)})
        print(f"world_size={world_size}: {records[-1]['status']}")
    if any(record["status"] == "failed" for record in records):
        status, correctness = "failed", {"status": "failed"}
    elif any(record["status"] == "success" for record in records):
        status, correctness = "success", {"status": "passed"}
    else:
        status, correctness = "unavailable", {"status": "not_run"}
    payload = build_artifact(
        artifact_type="module04.ddp_bridge",
        repository_root=REPOSITORY_ROOT,
        environment=environment,
        config=config,
        measurement={
            "profiler": "torch.profiler CPU+CUDA",
            "profile_steps": config["profile_steps"],
        },
        status=status,
        correctness=correctness,
        metrics={"records": records},
        raw_artifacts=raw_artifacts,
        boundary=(
            "The bridge explains the recorded module 03 MLP on this host. It is not a "
            "production training benchmark or a cross-host performance ranking."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(1 if status == "failed" else 0)


if __name__ == "__main__":
    main()

