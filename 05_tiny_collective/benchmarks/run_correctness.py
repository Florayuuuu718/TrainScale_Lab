"""Run the complete CPU/Gloo TinyCollective correctness matrix."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path

import torch

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks"))

from artifact_contract import build_artifact  # noqa: E402
from benchmarks.launcher import launch  # noqa: E402
from trainscale_collective.contract import load_correctness_config  # noqa: E402
from trainscale_collective.schedule import ring_volume  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_correctness_config(args.config)
    records = []
    with tempfile.TemporaryDirectory(prefix="trainscale-05-") as temporary:
        root = Path(temporary)
        for world_size in config["world_sizes"]:
            for elements in config["element_counts"]:
                for algorithm in config["algorithms"]:
                    case_id = f"{algorithm}-w{world_size}-n{elements}"
                    job = launch(
                        world_size=world_size,
                        rank_directory=root / case_id,
                        worker_args=[
                            "--mode",
                            "correctness",
                            "--algorithm",
                            algorithm,
                            "--backend",
                            "gloo",
                            "--device",
                            "cpu",
                            "--dtype",
                            config["dtype"],
                            "--elements",
                            str(elements),
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
                    passed = job["status"] == "success" and all(
                        rank["correctness_passed"] for rank in job["ranks"]
                    )
                    if algorithm == "ring":
                        expected_events = {rank: 2 * (world_size - 1) for rank in range(world_size)}
                    else:
                        expected_events = {
                            rank: 2 * (world_size - 1) if rank == 0 else 2
                            for rank in range(world_size)
                        }
                    trace_ok = all(
                        len(rank["trace"]) == expected_events[rank["rank"]] for rank in job["ranks"]
                    )
                    records.append(
                        {
                            "case_id": case_id,
                            "algorithm": algorithm,
                            "world_size": world_size,
                            "elements": elements,
                            "status": "success" if passed and trace_ok else "failed",
                            "trace_events_per_rank": expected_events,
                            "ring_volume": ring_volume(elements, world_size)
                            if algorithm == "ring"
                            else None,
                            "ranks": job["ranks"],
                            "stderr_tail": job["stderr_tail"],
                        }
                    )
                    print(f"{case_id}: {records[-1]['status']}")
    success = all(record["status"] == "success" for record in records)
    payload = build_artifact(
        artifact_type="module05.correctness",
        repository_root=REPOSITORY_ROOT,
        environment={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "backend": "gloo",
            "device": "cpu",
        },
        config=config,
        measurement={"case_count": len(records), "reference": "torch.distributed.all_reduce"},
        status="success" if success else "failed",
        correctness={"status": "passed" if success else "failed"},
        metrics={"records": records},
        raw_artifacts=[],
        boundary="CPU/Gloo proves schedule and math, not GPU/NCCL performance.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
