"""Profile two CPU/Gloo DDP ranks and archive communication-related rows and traces."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_distributed.contract import load_correctness_config  # noqa: E402

from benchmarks.launcher import launch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=MODULE_ROOT / "configs" / "correctness.toml",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--trace-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_correctness_config(args.config)
    args.trace_directory.mkdir(parents=True, exist_ok=True)
    worker_args = [
        "--mode",
        "profile",
        "--backend",
        "gloo",
        "--device",
        "cpu",
        "--seed",
        str(config["seed"]),
        "--input-dim",
        str(config["input_dim"]),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--num-classes",
        str(config["num_classes"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--profile-steps",
        str(args.steps),
        "--trace-directory",
        str(args.trace_directory.resolve()),
    ]
    with tempfile.TemporaryDirectory(prefix="trainscale-03-profile-") as temporary:
        job = launch(
            world_size=args.world_size,
            rank_directory=Path(temporary) / "ranks",
            worker_args=worker_args,
            timeout_seconds=args.timeout_seconds,
        )
    passed = job["status"] == "success" and all(
        rank["distributed_rows"] for rank in job["ranks"]
    )
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(),
        "scope": "CPU/Gloo DDP communication profile",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "world_size": args.world_size,
        "profile_steps": args.steps,
        "ranks": job["ranks"],
        "all_checks_passed": passed,
        "aggregation_note": (
            "Profiler key_averages rows can be nested; do not sum them as wall time."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"ranks={len(job['ranks'])}")
    print(args.output)
    print(f"all_checks_passed={passed}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
