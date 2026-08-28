"""Run and aggregate the fixed five-repetition long-window DDP scaling follow-up."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
MODULE03_RUNNER = (
    REPOSITORY_ROOT / "03_distributed_training" / "benchmarks" / "run_scaling.py"
)
AGGREGATOR = MODULE_ROOT / "benchmarks" / "aggregate_ddp_scaling.py"


def load_aggregator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("module04_scaling_aggregate", AGGREGATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {AGGREGATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scaling_command(config: Path, output: Path, timeout_seconds: int) -> list[str]:
    return [
        sys.executable,
        str(MODULE03_RUNNER),
        "--config",
        str(config),
        "--output",
        str(output),
        "--timeout-seconds",
        str(timeout_seconds),
    ]


def load_campaign_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if set(payload) != {"benchmark"} or not isinstance(payload["benchmark"], dict):
        raise ValueError(f"{path}: expected only a [benchmark] table")
    return dict(payload["benchmark"])


def validate_campaign_config(config: dict[str, Any]) -> None:
    if config["world_sizes"] != [1, 2, 4] or config["modes"] != ["strong", "weak"]:
        raise ValueError("campaign requires strong/weak world sizes 1/2/4")
    if config["warmup_steps"] < 200 or config["measured_steps"] < 5000:
        raise ValueError("campaign requires at least 200 warm-up and 5000 measured steps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--stability-threshold", type=float, default=0.05)
    parser.add_argument("--warning-threshold", type=float, default=0.10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repetitions != 5:
        raise ValueError("the formal long-window campaign requires exactly five repetitions")
    config = load_campaign_config(args.config)
    validate_campaign_config(config)
    if args.dry_run:
        for repetition in range(1, args.repetitions + 1):
            output = args.output_directory / f"run{repetition}.json"
            print(" ".join(scaling_command(args.config, output, args.timeout_seconds)))
        print(f"summary={args.summary_output}")
        return
    args.output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for repetition in range(1, args.repetitions + 1):
        output = args.output_directory / f"run{repetition}.json"
        command = scaling_command(args.config, output, args.timeout_seconds)
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=False,
            text=True,
            timeout=args.timeout_seconds,
        )
        (args.output_directory / f"run{repetition}.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (args.output_directory / f"run{repetition}.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        print(f"repetition={repetition} returncode={completed.returncode}")
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError(f"DDP scaling repetition {repetition} failed")
        paths.append(output)
    aggregator = load_aggregator()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    payload = aggregator.aggregate_sources(
        payloads,
        stability_threshold=args.stability_threshold,
        warning_threshold=args.warning_threshold,
        expected_repetitions=args.repetitions,
    )
    payload["source_artifacts"] = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": aggregator.sha256_file(path),
        }
        for path in paths
    ]
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    quality = payload["measurement_quality"]["status"]
    print(args.summary_output)
    print(f"measurement_quality={quality}")
    raise SystemExit(0 if quality == "passed" else 2)


if __name__ == "__main__":
    main()
