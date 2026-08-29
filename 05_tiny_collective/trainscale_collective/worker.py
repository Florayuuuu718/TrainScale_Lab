"""Torchrun worker for TinyCollective correctness and performance cases."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from trainscale_collective.algorithms import centralized_all_reduce, ring_all_reduce

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def input_tensor(
    elements: int, dtype: torch.dtype, device: torch.device, rank: int
) -> torch.Tensor:
    base = torch.arange(elements, dtype=torch.float32, device=device)
    return (base / max(elements, 1) + rank * 0.25).to(dtype)


def reference_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    result = tensor.clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def algorithm(name: str) -> Callable[[torch.Tensor], tuple[torch.Tensor, list[dict[str, Any]]]]:
    if name == "centralized":
        return centralized_all_reduce
    if name == "ring":
        return ring_all_reduce
    raise ValueError(f"unsupported TinyCollective algorithm: {name}")


def run_correctness(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    source = input_tensor(args.elements, DTYPE_MAP[args.dtype], device, dist.get_rank())
    expected = reference_all_reduce(source)
    actual, trace = algorithm(args.algorithm)(source.clone())
    error = float((actual.float() - expected.float()).abs().max().item())
    passed = torch.allclose(actual.float(), expected.float(), atol=args.atol, rtol=args.rtol)
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "algorithm": args.algorithm,
        "elements": args.elements,
        "dtype": args.dtype,
        "max_error": error,
        "correctness_passed": bool(passed),
        "trace": trace,
    }


def _one_collective(name: str, source: torch.Tensor) -> torch.Tensor:
    if name == "torch":
        return reference_all_reduce(source)
    result, _ = algorithm(name)(source)
    return result


def run_benchmark(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    rows = []
    for message_bytes in args.message_bytes:
        element_size = torch.empty((), dtype=DTYPE_MAP[args.dtype]).element_size()
        if message_bytes % element_size:
            raise ValueError("message bytes must be divisible by dtype element size")
        elements = message_bytes // element_size
        source = input_tensor(elements, DTYPE_MAP[args.dtype], device, dist.get_rank())
        expected = reference_all_reduce(source)
        for _ in range(args.warmup_iterations):
            _one_collective(args.algorithm, source.clone())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dist.barrier()
        start = time.perf_counter()
        actual = source
        for _ in range(args.measured_iterations):
            actual = _one_collective(args.algorithm, source.clone())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        maximum = torch.tensor(elapsed, dtype=torch.float64, device=device)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        elapsed = float(maximum.item())
        latency_us = elapsed * 1e6 / args.measured_iterations
        algbw = message_bytes / (latency_us * 1e-6) / 1e9
        busbw = algbw * 2 * (dist.get_world_size() - 1) / dist.get_world_size()
        error = float((actual.float() - expected.float()).abs().max().item())
        rows.append(
            {
                "message_bytes": message_bytes,
                "elements": elements,
                "latency_us": latency_us,
                "algbw_gbps": algbw,
                "busbw_gbps": busbw,
                "max_error": error,
                "correctness_passed": bool(
                    torch.allclose(actual.float(), expected.float(), atol=args.atol, rtol=args.rtol)
                ),
            }
        )
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "algorithm": args.algorithm,
        "dtype": args.dtype,
        "warmup_iterations": args.warmup_iterations,
        "measured_iterations": args.measured_iterations,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("correctness", "benchmark"), required=True)
    parser.add_argument("--algorithm", choices=("centralized", "ring", "torch"), required=True)
    parser.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPE_MAP), required=True)
    parser.add_argument("--rank-directory", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=1)
    parser.add_argument("--message-bytes", type=int, nargs="+", default=[8])
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--measured-iterations", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed + int(os.environ.get("RANK", "0")))
    if args.device == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        torch.set_num_threads(1)
        device = torch.device("cpu")
    dist.init_process_group(backend=args.backend, timeout=timedelta(seconds=args.timeout_seconds))
    try:
        payload = (
            run_correctness(args, device)
            if args.mode == "correctness"
            else run_benchmark(args, device)
        )
        payload["status"] = "success"
        args.rank_directory.mkdir(parents=True, exist_ok=True)
        path = args.rank_directory / f"rank_{dist.get_rank()}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
