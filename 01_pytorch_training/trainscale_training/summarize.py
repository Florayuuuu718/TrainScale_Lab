"""Collect reproducible module 01 artifacts into one compact tracked summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .reporting import write_json


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_summary(path: Path) -> dict[str, Any]:
    value = _read(path)
    final = value["history"][-1]
    return {
        "experiment_name": value["experiment_name"],
        "total_wall_seconds": value["total_wall_seconds"],
        "global_step": value["global_step"],
        "peak_cuda_memory_bytes": value["peak_cuda_memory_bytes"],
        "final_epoch": final["epoch"],
        "final_train_loss": final["train"]["loss"],
        "final_train_accuracy": final["train"]["accuracy"],
        "final_valid_loss": final["valid"]["loss"],
        "final_valid_accuracy": final["valid"]["accuracy"],
        "final_train_samples_per_second": final["train"]["samples_per_second"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="01_pytorch_training/results")
    parser.add_argument("--output", default="01_pytorch_training/results/summary.json")
    args = parser.parse_args()
    root = Path(args.root)
    raw = root / "raw"
    training_runs = {}
    for name in ("synthetic_cpu", "synthetic_cuda", "synthetic_accumulation", "cifar10_baseline"):
        path = raw / name / "summary.json"
        if path.is_file():
            training_runs[name] = _training_summary(path)
    value = {
        "training_runs": training_runs,
        "ablations": {
            "synthetic": _read(root / "synthetic_ablation.json"),
            "cifar10_long_wsl": _read(root / "cifar10_modes_wsl.json"),
        },
        "dataloader_workers": {
            "synthetic_short": _read(root / "dataloader_workers.json"),
            "jpeg_long": _read(root / "dataloader_image_workers.json"),
        },
        "profiler": {
            "cpu_activity": _read(root / "profiler_summary.json"),
            "cuda_activity": _read(root / "cifar10_cuda_profiler_wsl_cu129.json"),
        },
    }
    write_json(args.output, value)


if __name__ == "__main__":
    main()
