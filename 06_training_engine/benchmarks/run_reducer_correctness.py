"""Run the CPU/Gloo global-batch reducer-equivalence matrix for Module 06."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks"))

from artifact_contract import build_artifact, sha256_file  # noqa: E402
from torchrun_launcher import launch_torchrun  # noqa: E402
from trainscale_engine.contract import load_correctness_config  # noqa: E402

WORKER = MODULE_ROOT / "trainscale_engine" / "worker.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_correctness_config(args.config)
    args.raw_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for world_size in config["world_sizes"]:
        for strategy in config["strategies"]:
            for accumulation_steps in config["accumulation_steps"]:
                case_id = f"{strategy}-w{world_size}-a{accumulation_steps}"
                job = launch_torchrun(
                    repository_root=REPOSITORY_ROOT,
                    worker=WORKER,
                    world_size=world_size,
                    rank_directory=args.raw_directory / case_id,
                    worker_args=[
                        "--mode",
                        "correctness",
                        "--strategy",
                        strategy,
                        "--backend",
                        "gloo",
                        "--device",
                        "cpu",
                        "--model-preset",
                        config["model_preset"],
                        *(["--include-unused"] if config["include_unused"] else []),
                        "--global-batch-size",
                        str(config["global_batch_size"]),
                        "--accumulation-steps",
                        str(accumulation_steps),
                        "--bucket-cap-bytes",
                        str(config["bucket_cap_bytes"]),
                        "--learning-rate",
                        str(config["learning_rate"]),
                        "--seed",
                        str(config["seed"]),
                        "--atol",
                        str(config["atol"]),
                        "--rtol",
                        str(config["rtol"]),
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
                        "status": "success" if passed else "failed",
                        "ranks": job["ranks"],
                        "stderr_tail": job["stderr_tail"],
                    }
                )
                print(f"{case_id}: {records[-1]['status']}")
    success = all(record["status"] == "success" for record in records)
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
        artifact_type="module06.local_correctness",
        repository_root=REPOSITORY_ROOT,
        environment={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "backend": "gloo",
            "device": "cpu",
        },
        config=config,
        measurement={
            "case_count": len(records),
            "reference": "single-process global-batch update",
        },
        status="success" if success else "failed",
        correctness={"status": "passed" if success else "failed"},
        metrics={"records": records},
        raw_artifacts=raw_artifacts,
        boundary="CPU/Gloo validates reducer math and lifecycle, not CUDA/NCCL overlap.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
