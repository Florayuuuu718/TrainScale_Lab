"""One torchrun worker used by every module 03 correctness and scaling experiment."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def make_model(input_dim: int, hidden_dim: int, num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )


def make_dataset(
    size: int, input_dim: int, num_classes: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(size, input_dim, generator=generator)
    rule = torch.randn(input_dim, num_classes, generator=generator)
    labels = (features @ rule).argmax(dim=1)
    return features, labels


def parameter_vector(model: nn.Module) -> list[float]:
    flattened = [parameter.detach().cpu().flatten() for parameter in model.parameters()]
    return torch.cat(flattened).tolist()


def write_rank_result(rank_directory: Path, rank: int, payload: dict[str, Any]) -> None:
    path = rank_directory / f"rank_{rank}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def wrap_ddp(model: nn.Module, device: torch.device, local_rank: int) -> DDP:
    if device.type == "cuda":
        return DDP(model, device_ids=[local_rank], output_device=local_rank)
    return DDP(model)


def run_semantics(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    rank = dist.get_rank()
    rank_value = torch.tensor(float(rank), device=device)
    dist.all_reduce(rank_value, op=dist.ReduceOp.SUM)
    broadcast_value = torch.tensor(42.0 if rank == 0 else -1.0, device=device)
    dist.broadcast(broadcast_value, src=0)
    dist.barrier()
    return {
        "rank": rank,
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "world_size": dist.get_world_size(),
        "backend": dist.get_backend(),
        "hostname": socket.gethostname(),
        "all_reduce_rank_sum": rank_value.item(),
        "broadcast_value": broadcast_value.item(),
    }


def run_sampler(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    del device
    dataset = TensorDataset(torch.arange(args.dataset_size))
    sampler: DistributedSampler[Any] = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    sampler.set_epoch(args.epoch)
    indices = list(iter(sampler))
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "epoch": args.epoch,
        "indices": indices,
    }


def run_gradient(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.global_batch_size % world_size != 0:
        raise ValueError("global_batch_size must be divisible by world_size")
    torch.manual_seed(args.seed)
    model = make_model(args.input_dim, args.hidden_dim, args.num_classes).to(device)
    reference = make_model(args.input_dim, args.hidden_dim, args.num_classes).to(device)
    reference.load_state_dict(model.state_dict())
    ddp = wrap_ddp(model, device, int(os.environ["LOCAL_RANK"]))
    optimizer = torch.optim.SGD(ddp.parameters(), lr=args.learning_rate)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=args.learning_rate)
    features, labels = make_dataset(
        args.global_batch_size, args.input_dim, args.num_classes, args.seed + 1
    )
    features = features.to(device)
    labels = labels.to(device)
    local_batch = args.global_batch_size // world_size
    start = rank * local_batch
    stop = start + local_batch

    optimizer.zero_grad(set_to_none=True)
    local_loss = nn.functional.cross_entropy(ddp(features[start:stop]), labels[start:stop])
    local_loss.backward()

    gradient_max_error: float | None = None
    parameter_max_error: float | None = None
    if rank == 0:
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = nn.functional.cross_entropy(reference(features), labels)
        reference_loss.backward()
        errors = [
            (actual.grad - expected.grad).abs().max().item()
            for actual, expected in zip(
                ddp.module.parameters(), reference.parameters(), strict=True
            )
            if actual.grad is not None and expected.grad is not None
        ]
        gradient_max_error = max(errors, default=0.0)

    optimizer.step()
    if rank == 0:
        reference_optimizer.step()
        parameter_max_error = max(
            (actual - expected).abs().max().item()
            for actual, expected in zip(
                ddp.module.parameters(), reference.parameters(), strict=True
            )
        )
    dist.barrier()
    return {
        "rank": rank,
        "world_size": world_size,
        "local_batch_size": local_batch,
        "local_loss": local_loss.item(),
        "gradient_max_error_vs_global_batch": gradient_max_error,
        "parameter_max_error_vs_global_batch": parameter_max_error,
        "parameter_vector": parameter_vector(ddp.module),
    }


def run_train(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.global_batch_size % world_size != 0:
        raise ValueError("global_batch_size must be divisible by world_size")
    torch.manual_seed(args.seed)
    features, labels = make_dataset(
        args.dataset_size, args.input_dim, args.num_classes, args.seed + 11
    )
    dataset = TensorDataset(features, labels)
    sampler: DistributedSampler[Any] = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    local_batch = args.global_batch_size // world_size
    loader = DataLoader(dataset, batch_size=local_batch, sampler=sampler, num_workers=0)
    model = make_model(args.input_dim, args.hidden_dim, args.num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["next_epoch"])
    ddp = wrap_ddp(model, device, int(os.environ["LOCAL_RANK"]))
    history: list[dict[str, float | int]] = []
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        loss_sum = torch.zeros((), device=device)
        sample_count = torch.zeros((), device=device)
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(ddp(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * batch_labels.numel()
            sample_count += batch_labels.numel()
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
        history.append({"epoch": epoch, "global_mean_loss": (loss_sum / sample_count).item()})

    dist.barrier()
    if rank == 0 and args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": ddp.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "next_epoch": args.epochs,
                "seed": args.seed,
            },
            args.checkpoint,
        )
    dist.barrier()
    return {
        "rank": rank,
        "world_size": world_size,
        "start_epoch": start_epoch,
        "target_epochs": args.epochs,
        "local_batch_size": local_batch,
        "history": history,
        "parameter_vector": parameter_vector(ddp.module),
        "checkpoint_writer": rank == 0 and args.checkpoint is not None,
    }


def benchmark_step(
    ddp: DDP,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(ddp(features), labels)
    loss.backward()
    optimizer.step()


def run_benchmark(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    world_size = dist.get_world_size()
    if args.scaling_mode == "strong":
        if args.global_batch_size % world_size != 0:
            raise ValueError("global_batch_size must be divisible by world_size")
        local_batch = args.global_batch_size // world_size
    else:
        local_batch = args.per_rank_batch_size
    global_batch = local_batch * world_size
    torch.manual_seed(args.seed)
    model = make_model(args.input_dim, args.hidden_dim, args.num_classes).to(device)
    ddp = wrap_ddp(model, device, int(os.environ["LOCAL_RANK"]))
    optimizer = torch.optim.SGD(ddp.parameters(), lr=args.learning_rate)
    features, labels = make_dataset(
        local_batch, args.input_dim, args.num_classes, args.seed + 101 + dist.get_rank()
    )
    features = features.to(device)
    labels = labels.to(device)
    for _ in range(args.warmup_steps):
        benchmark_step(ddp, optimizer, features, labels)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    start = time.perf_counter()
    for _ in range(args.measured_steps):
        benchmark_step(ddp, optimizer, features, labels)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    maximum_elapsed = torch.tensor(elapsed, device=device)
    dist.all_reduce(maximum_elapsed, op=dist.ReduceOp.MAX)
    elapsed = maximum_elapsed.item()
    throughput = global_batch * args.measured_steps / elapsed
    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    return {
        "rank": dist.get_rank(),
        "world_size": world_size,
        "mode": args.scaling_mode,
        "backend": dist.get_backend(),
        "device": device.type,
        "local_batch_size": local_batch,
        "global_batch_size": global_batch,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "max_rank_elapsed_seconds": elapsed,
        "global_samples_per_second": throughput,
        "peak_memory_allocated_bytes": peak_memory,
    }


def run_profile(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if device.type != "cpu":
        raise ValueError("the teaching profile currently records CPU/Gloo only")
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    torch.manual_seed(args.seed)
    model = make_model(args.input_dim, args.hidden_dim, args.num_classes).to(device)
    ddp = wrap_ddp(model, device, int(os.environ["LOCAL_RANK"]))
    optimizer = torch.optim.SGD(ddp.parameters(), lr=args.learning_rate)
    features, labels = make_dataset(32, args.input_dim, args.num_classes, args.seed + 301)
    features = features.to(device)
    labels = labels.to(device)
    for _ in range(2):
        benchmark_step(ddp, optimizer, features, labels)
    dist.barrier()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        for _ in range(args.profile_steps):
            benchmark_step(ddp, optimizer, features, labels)
    trace_path = args.trace_directory / f"rank_{dist.get_rank()}_trace.json"
    args.trace_directory.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    distributed_rows = []
    for row in prof.key_averages():
        lowered = row.key.lower()
        if any(token in lowered for token in ("allreduce", "c10d", "gloo", "distributed")):
            distributed_rows.append(
                {
                    "name": row.key,
                    "count": row.count,
                    "cpu_time_total_us": row.cpu_time_total,
                }
            )
    distributed_rows.sort(key=lambda row: row["cpu_time_total_us"], reverse=True)
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "profile_steps": args.profile_steps,
        "distributed_rows": distributed_rows[:20],
        "trace": str(trace_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("semantics", "sampler", "gradient", "train", "benchmark", "profile"),
        required=True,
    )
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rank-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--dataset-size", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--per-rank-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=12)
    parser.add_argument("--scaling-mode", choices=("strong", "weak"), default="strong")
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--trace-directory", type=Path, default=Path("/tmp/trainscale-ddp"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        torch.set_num_threads(1)
        device = torch.device("cpu")
    dist.init_process_group(backend=args.backend)
    try:
        function = {
            "semantics": run_semantics,
            "sampler": run_sampler,
            "gradient": run_gradient,
            "train": run_train,
            "benchmark": run_benchmark,
            "profile": run_profile,
        }[args.mode]
        payload = function(args, device)
        payload["status"] = "success"
        payload["worker_mode"] = args.mode
        payload.setdefault("mode", args.mode)
        write_rank_result(args.rank_directory, dist.get_rank(), payload)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
