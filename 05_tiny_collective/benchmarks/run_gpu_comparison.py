"""Compare centralized, ring, and torch.distributed AllReduce on 2/4 GPUs."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks"))

from artifact_contract import build_artifact, sha256_file  # noqa: E402
from benchmarks.launcher import launch  # noqa: E402
from trainscale_collective.contract import load_benchmark_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_benchmark_config(args.config)
    available = torch.cuda.device_count()
    environment_ready = (
        torch.cuda.is_available()
        and dist.is_nccl_available()
        and available >= max(config["world_sizes"])
    )
    records = []
    args.raw_directory.mkdir(parents=True, exist_ok=True)
    for world_size in config["world_sizes"]:
        for algorithm in config["algorithms"]:
            if not environment_ready:
                records.append(
                    {"algorithm": algorithm, "world_size": world_size, "status": "unavailable"}
                )
                continue
            repetitions = []
            for repetition in range(1, config["repetitions"] + 1):
                job = launch(
                    world_size=world_size,
                    rank_directory=args.raw_directory / f"{algorithm}-w{world_size}-r{repetition}",
                    worker_args=[
                        "--mode",
                        "benchmark",
                        "--algorithm",
                        algorithm,
                        "--backend",
                        "nccl",
                        "--device",
                        "cuda",
                        "--dtype",
                        config["dtype"],
                        "--message-bytes",
                        *[str(value) for value in config["message_bytes"]],
                        "--warmup-iterations",
                        str(config["warmup_iterations"]),
                        "--measured-iterations",
                        str(config["measured_iterations"]),
                        "--atol",
                        str(config["atol"]),
                        "--rtol",
                        str(config["rtol"]),
                        "--timeout-seconds",
                        str(config["timeout_seconds"]),
                        "--seed",
                        str(config["seed"]),
                    ],
                    timeout_seconds=config["timeout_seconds"],
                )
                if job["status"] != "success" or not all(
                    all(row["correctness_passed"] for row in rank["rows"]) for rank in job["ranks"]
                ):
                    repetitions.append(
                        {
                            "status": "failed",
                            "command": job["command"],
                            "returncode": job["returncode"],
                            "timed_out": job["timed_out"],
                            "stderr_tail": job["stderr_tail"],
                        }
                    )
                else:
                    repetitions.append(
                        {
                            "status": "success",
                            "command": job["command"],
                            "rows": job["ranks"][0]["rows"],
                        }
                    )
            success = all(item["status"] == "success" for item in repetitions)
            rows = []
            if success:
                for index, message_bytes in enumerate(config["message_bytes"]):
                    samples = [item["rows"][index] for item in repetitions]
                    latencies = [sample["latency_us"] for sample in samples]
                    bandwidths = [sample["busbw_gbps"] for sample in samples]
                    median_latency = statistics.median(latencies)
                    median_bandwidth = statistics.median(bandwidths)
                    rows.append(
                        {
                            "message_bytes": message_bytes,
                            "median_latency_us": median_latency,
                            "latency_relative_range": (max(latencies) - min(latencies))
                            / median_latency,
                            "median_busbw_gbps": median_bandwidth,
                            "busbw_relative_range": (max(bandwidths) - min(bandwidths))
                            / median_bandwidth,
                            "raw_repetitions": samples,
                        }
                    )
            records.append(
                {
                    "algorithm": algorithm,
                    "world_size": world_size,
                    "status": "success" if success else "failed",
                    "rows": rows,
                    "repetitions": repetitions,
                }
            )
            print(f"{algorithm}/world_size={world_size}: {records[-1]['status']}")
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
        artifact_type="module05.gpu_comparison",
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
        config=config,
        measurement={
            "aggregation": f"median of {config['repetitions']} independent jobs",
            "variability": "(maximum - minimum) / median",
            "timer": "slowest rank wall time",
        },
        status=status,
        correctness={"status": "failed" if failed else "passed" if succeeded else "not_run"},
        metrics={"records": records},
        raw_artifacts=raw_artifacts,
        boundary="Python P2P ring is a teaching implementation, not an NCCL replacement.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
