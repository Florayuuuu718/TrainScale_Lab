"""Profile the module 03 MLP with DDP/NCCL and write one result per rank."""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile


def make_model(input_dim: int, hidden_dim: int, num_classes: int) -> nn.Module:
    """Match the model used by module 03 scaling exactly."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )


def train_step(
    ddp: DDP,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(ddp(features), labels)
    loss.backward()
    optimizer.step()


def communication_events(profiler: Any) -> list[dict[str, Any]]:
    """Keep communication-related aggregates while retaining raw traces for audit."""
    selected: list[dict[str, Any]] = []
    for event in profiler.key_averages():
        name = str(event.key)
        normalized = name.lower().replace("_", "")
        if not any(token in normalized for token in ("allreduce", "nccl", "c10d")):
            continue
        selected.append(
            {
                "name": name,
                "count": event.count,
                "cpu_time_total_us": event.cpu_time_total,
                "device_time_total_us": getattr(event, "device_time_total", 0.0),
            }
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-directory", type=Path, required=True)
    parser.add_argument("--trace-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--per-rank-batch-size", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--profile-steps", type=int, required=True)
    parser.add_argument("--bucket-cap-mb", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=args.timeout_seconds))
    try:
        torch.manual_seed(args.seed)
        model = make_model(args.input_dim, args.hidden_dim, args.num_classes).to(device)
        ddp = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            bucket_cap_mb=args.bucket_cap_mb,
        )
        optimizer = torch.optim.SGD(ddp.parameters(), lr=args.learning_rate)
        generator = torch.Generator(device=device).manual_seed(args.seed + 101 + rank)
        features = torch.randn(
            args.per_rank_batch_size,
            args.input_dim,
            generator=generator,
            device=device,
        )
        labels = torch.randint(
            args.num_classes,
            (args.per_rank_batch_size,),
            generator=generator,
            device=device,
        )
        for _ in range(args.warmup_steps):
            train_step(ddp, optimizer, features, labels)
        dist.barrier()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profiler:
            for _ in range(args.profile_steps):
                train_step(ddp, optimizer, features, labels)
        torch.cuda.synchronize(device)
        args.trace_directory.mkdir(parents=True, exist_ok=True)
        trace_path = args.trace_directory / f"rank_{rank}_trace.json"
        profiler.export_chrome_trace(str(trace_path))
        parameter_sums = [
            parameter.detach().float().sum() for parameter in model.parameters()
        ]
        checksum = torch.stack(parameter_sums).sum()
        minimum = checksum.clone()
        maximum = checksum.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        max_checksum_error = float((maximum - minimum).abs().item())
        logging_data = ddp._get_ddp_logging_data()  # noqa: SLF001
        result = {
            "rank": rank,
            "world_size": dist.get_world_size(),
            "status": "success",
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "fp32_gradient_payload_bytes": sum(
                parameter.numel() * parameter.element_size() for parameter in model.parameters()
            ),
            "configured_bucket_cap_mb": args.bucket_cap_mb,
            "ddp_logging_data": {
                key: value
                for key, value in logging_data.items()
                if key in {"bucket_sizes", "bucket_cap_bytes", "num_parameter_tensors"}
            },
            "profile_steps": args.profile_steps,
            "communication_events": communication_events(profiler),
            "max_parameter_checksum_error": max_checksum_error,
            "parameters_consistent": max_checksum_error <= 1e-6,
            "trace": str(trace_path),
        }
        args.rank_directory.mkdir(parents=True, exist_ok=True)
        result_path = args.rank_directory / f"rank_{rank}.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
