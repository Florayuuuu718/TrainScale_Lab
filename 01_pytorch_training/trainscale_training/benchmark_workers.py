"""Measure DataLoader throughput while changing only num_workers."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import DelayedSyntheticDataset


def measure(num_workers: int, args: argparse.Namespace) -> dict[str, float | int]:
    rates: list[float] = []
    for _ in range(args.repeats):
        dataset = DelayedSyntheticDataset(args.samples, args.input_dim, args.delay_ms, args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers)
        started = time.perf_counter()
        seen = sum(features.shape[0] for features, _ in loader)
        rates.append(seen / (time.perf_counter() - started))
    return {
        "num_workers": num_workers,
        "samples": args.samples,
        "median_samples_per_second": statistics.median(rates),
        "min_samples_per_second": min(rates),
        "max_samples_per_second": max(rates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--delay-ms", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = {
        "experiment": "dataloader_workers_throughput",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "controlled_variables": {
            "samples": args.samples,
            "input_dim": args.input_dim,
            "batch_size": args.batch_size,
            "delay_ms": args.delay_ms,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "results": [measure(workers, args) for workers in args.workers],
    }
    output = json.dumps(value, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
