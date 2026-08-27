"""Run one crash-isolated Gloo/DDP correctness experiment through torchrun."""

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
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_distributed.contract import load_correctness_config  # noqa: E402

from benchmarks.launcher import launch  # noqa: E402


def environment_manifest() -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gloo_available": dist.is_gloo_available(),
        "cpu_count": __import__("os").cpu_count(),
    }


def base_worker_args(config: dict[str, Any]) -> list[str]:
    return [
        "--backend",
        "gloo",
        "--device",
        "cpu",
        "--seed",
        str(config["seed"]),
        "--dataset-size",
        str(config["dataset_size"]),
        "--input-dim",
        str(config["input_dim"]),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--num-classes",
        str(config["num_classes"]),
        "--global-batch-size",
        str(config["global_batch_size"]),
        "--learning-rate",
        str(config["learning_rate"]),
    ]


def maximum_vector_difference(vectors: list[list[float]]) -> float:
    if not vectors:
        return float("inf")
    reference = vectors[0]
    return max(
        (
            abs(actual - expected)
            for vector in vectors[1:]
            for actual, expected in zip(vector, reference, strict=True)
        ),
        default=0.0,
    )


def strip_vectors(ranks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in rank.items() if key != "parameter_vector"}
        for rank in ranks
    ]


def run_semantics(
    config: dict[str, Any], world_size: int, root: Path, timeout: int
) -> dict[str, Any]:
    job = launch(
        world_size=world_size,
        rank_directory=root / "semantics",
        worker_args=["--mode", "semantics", *base_worker_args(config)],
        timeout_seconds=timeout,
    )
    expected_sum = world_size * (world_size - 1) / 2
    passed = job["status"] == "success" and all(
        rank["all_reduce_rank_sum"] == expected_sum
        and rank["broadcast_value"] == 42.0
        and rank["world_size"] == world_size
        for rank in job["ranks"]
    )
    return {
        "status": "success" if passed else "failed",
        "world_size": world_size,
        "expected_rank_sum": expected_sum,
        "ranks": job["ranks"],
        "launcher": {key: job[key] for key in ("command", "returncode", "stderr_tail")},
    }


def run_sampler(
    config: dict[str, Any], world_size: int, root: Path, timeout: int
) -> dict[str, Any]:
    from trainscale_distributed.contract import analyze_sampler_shards

    epochs = []
    for epoch in (0, 1):
        job = launch(
            world_size=world_size,
            rank_directory=root / f"sampler_epoch_{epoch}",
            worker_args=[
                "--mode",
                "sampler",
                *base_worker_args(config),
                "--epoch",
                str(epoch),
            ],
            timeout_seconds=timeout,
        )
        shards = [rank["indices"] for rank in sorted(job["ranks"], key=lambda row: row["rank"])]
        analysis = analyze_sampler_shards(shards, config["dataset_size"])
        epochs.append(
            {
                "epoch": epoch,
                "status": job["status"],
                "shards": shards,
                "analysis": analysis,
            }
        )
    order_changed = epochs[0]["shards"] != epochs[1]["shards"]
    passed = all(
        epoch["status"] == "success" and epoch["analysis"]["coverage_complete"]
        for epoch in epochs
    ) and order_changed
    return {
        "status": "success" if passed else "failed",
        "world_size": world_size,
        "set_epoch_changed_order": order_changed,
        "epochs": epochs,
    }


def run_gradient(
    config: dict[str, Any], world_size: int, root: Path, timeout: int
) -> dict[str, Any]:
    job = launch(
        world_size=world_size,
        rank_directory=root / "gradient",
        worker_args=["--mode", "gradient", *base_worker_args(config)],
        timeout_seconds=timeout,
    )
    rank_zero: dict[str, Any] = next(
        (rank for rank in job["ranks"] if rank["rank"] == 0), {}
    )
    vectors = [rank["parameter_vector"] for rank in job["ranks"]]
    consistency = maximum_vector_difference(vectors)
    gradient_error = rank_zero.get("gradient_max_error_vs_global_batch", float("inf"))
    parameter_error = rank_zero.get("parameter_max_error_vs_global_batch", float("inf"))
    passed = (
        job["status"] == "success"
        and consistency <= 1e-7
        and gradient_error <= 1e-6
        and parameter_error <= 1e-7
    )
    return {
        "status": "success" if passed else "failed",
        "world_size": world_size,
        "rank_parameter_max_difference": consistency,
        "gradient_max_error_vs_global_batch": gradient_error,
        "parameter_max_error_vs_global_batch": parameter_error,
        "ranks": strip_vectors(job["ranks"]),
        "launcher": {key: job[key] for key in ("command", "returncode", "stderr_tail")},
    }


def run_checkpoint(
    config: dict[str, Any], world_size: int, root: Path, timeout: int
) -> dict[str, Any]:
    continuous_checkpoint = root / "continuous.pt"
    partial_checkpoint = root / "partial.pt"
    resumed_checkpoint = root / "resumed.pt"
    common = base_worker_args(config)
    continuous = launch(
        world_size=world_size,
        rank_directory=root / "continuous",
        worker_args=[
            "--mode",
            "train",
            *common,
            "--epochs",
            str(config["epochs"]),
            "--checkpoint",
            str(continuous_checkpoint),
        ],
        timeout_seconds=timeout,
    )
    halfway = config["epochs"] // 2
    partial = launch(
        world_size=world_size,
        rank_directory=root / "partial",
        worker_args=[
            "--mode",
            "train",
            *common,
            "--epochs",
            str(halfway),
            "--checkpoint",
            str(partial_checkpoint),
        ],
        timeout_seconds=timeout,
    )
    resumed = launch(
        world_size=world_size,
        rank_directory=root / "resumed",
        worker_args=[
            "--mode",
            "train",
            *common,
            "--epochs",
            str(config["epochs"]),
            "--resume",
            str(partial_checkpoint),
            "--checkpoint",
            str(resumed_checkpoint),
        ],
        timeout_seconds=timeout,
    )
    continuous_vectors = [rank["parameter_vector"] for rank in continuous["ranks"]]
    resumed_vectors = [rank["parameter_vector"] for rank in resumed["ranks"]]
    final_error = max(
        (
            abs(actual - expected)
            for actual, expected in zip(
                resumed_vectors[0], continuous_vectors[0], strict=True
            )
        ),
        default=0.0,
    )
    writer_counts = {
        "continuous": sum(rank["checkpoint_writer"] for rank in continuous["ranks"]),
        "partial": sum(rank["checkpoint_writer"] for rank in partial["ranks"]),
        "resumed": sum(rank["checkpoint_writer"] for rank in resumed["ranks"]),
    }
    consistency = max(
        maximum_vector_difference(continuous_vectors),
        maximum_vector_difference(resumed_vectors),
    )
    passed = (
        all(job["status"] == "success" for job in (continuous, partial, resumed))
        and final_error <= 1e-7
        and consistency <= 1e-7
        and set(writer_counts.values()) == {1}
    )
    return {
        "status": "success" if passed else "failed",
        "world_size": world_size,
        "continuous_vs_resumed_parameter_max_error": final_error,
        "rank_parameter_max_difference": consistency,
        "checkpoint_writer_counts": writer_counts,
        "continuous_history": continuous["ranks"][0]["history"],
        "partial_history": partial["ranks"][0]["history"],
        "resumed_history": resumed["ranks"][0]["history"],
        "resume_start_epoch": resumed["ranks"][0]["start_epoch"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", choices=("semantics", "sampler", "gradient", "checkpoint"), required=True
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument(
        "--config",
        type=Path,
        default=MODULE_ROOT / "configs" / "correctness.toml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.world_size <= 0 or args.timeout_seconds <= 0:
        raise SystemExit("world-size and timeout must be positive")
    config = load_correctness_config(args.config)
    with tempfile.TemporaryDirectory(prefix=f"trainscale-03-{args.experiment}-") as temporary:
        result = {
            "semantics": run_semantics,
            "sampler": run_sampler,
            "gradient": run_gradient,
            "checkpoint": run_checkpoint,
        }[args.experiment](config, args.world_size, Path(temporary), args.timeout_seconds)
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(),
        "scope": args.experiment,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "environment": environment_manifest(),
        "config": config,
        "result": result,
        "all_checks_passed": result["status"] == "success",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{args.experiment}: {result['status']}")
    print(args.output)
    print(f"all_checks_passed={payload['all_checks_passed']}")
    raise SystemExit(0 if payload["all_checks_passed"] else 1)


if __name__ == "__main__":
    main()
