"""Torchrun worker for Module 06 correctness, performance, and timeline experiments."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from trainscale_engine.engine import ManualReducer, train_step
from trainscale_engine.model import TinyTransformer, make_classification_batch, model_preset
from trainscale_engine.reducer import BucketReducer, BulkReducer, PerParameterReducer


def _base_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _reducer(strategy: str, model: nn.Module, bucket_cap_bytes: int) -> ManualReducer | None:
    if strategy == "bulk":
        return BulkReducer(model)
    if strategy == "per_parameter":
        return PerParameterReducer(model)
    if strategy == "bucket_sync":
        return BucketReducer(model, bucket_cap_bytes, asynchronous=False)
    if strategy == "bucket_async":
        return BucketReducer(model, bucket_cap_bytes, asynchronous=True)
    if strategy == "ddp":
        return None
    raise ValueError(f"unsupported strategy: {strategy}")


def _model_and_reducer(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, ManualReducer | None]:
    torch.manual_seed(args.seed)
    base = TinyTransformer(model_preset(args.model_preset), include_unused=args.include_unused).to(
        device
    )
    if args.strategy == "ddp":
        local_rank = int(os.environ["LOCAL_RANK"])
        model: nn.Module = (
            DDP(
                base,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=args.include_unused,
            )
            if device.type == "cuda"
            else DDP(base, find_unused_parameters=args.include_unused)
        )
        return model, None
    return base, _reducer(args.strategy, base, args.bucket_cap_bytes)


def _gradients(model: nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _maximum_gradient_error(
    actual: dict[str, torch.Tensor | None], expected: dict[str, torch.Tensor | None]
) -> float:
    errors = []
    for name in expected:
        left, right = actual[name], expected[name]
        if left is None or right is None:
            if left is not None or right is not None:
                return float("inf")
            continue
        errors.append(float((left - right).abs().max()))
    return max(errors, default=0.0)


def _maximum_parameter_error(actual: nn.Module, expected: nn.Module) -> float:
    return max(
        float((left.detach() - right.detach()).abs().max())
        for left, right in zip(actual.parameters(), expected.parameters(), strict=True)
    )


def _gradients_close(
    actual: dict[str, torch.Tensor | None],
    expected: dict[str, torch.Tensor | None],
    *,
    atol: float,
    rtol: float,
) -> bool:
    for name, right in expected.items():
        left = actual[name]
        if left is None or right is None:
            if left is not None or right is not None:
                return False
        elif not torch.allclose(left, right, atol=atol, rtol=rtol):
            return False
    return True


def _parameters_close(actual: nn.Module, expected: nn.Module, *, atol: float, rtol: float) -> bool:
    return all(
        torch.allclose(left, right, atol=atol, rtol=rtol)
        for left, right in zip(actual.parameters(), expected.parameters(), strict=True)
    )


def run_correctness(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    world_size, rank = dist.get_world_size(), dist.get_rank()
    if args.global_batch_size % world_size:
        raise ValueError("global batch must be divisible by world size")
    local_batch = args.global_batch_size // world_size
    if local_batch % args.accumulation_steps:
        raise ValueError("local batch must be divisible by accumulation steps")
    config = model_preset(args.model_preset)
    tokens, labels = make_classification_batch(args.global_batch_size, config, args.seed + 1)
    tokens, labels = tokens.to(device), labels.to(device)

    torch.manual_seed(args.seed)
    reference = TinyTransformer(config, include_unused=args.include_unused).to(device)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=args.learning_rate)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_loss = nn.functional.cross_entropy(reference(tokens), labels)
    reference_loss.backward()
    expected_gradients = _gradients(reference)
    reference_optimizer.step()

    model, reducer = _model_and_reducer(args, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    captured: dict[str, torch.Tensor | None] = {}

    def capture(base: nn.Module) -> None:
        captured.update(_gradients(base))

    start, stop = rank * local_batch, (rank + 1) * local_batch
    result = train_step(
        model,
        optimizer,
        tokens[start:stop],
        labels[start:stop],
        reducer=reducer,
        accumulation_steps=args.accumulation_steps,
        before_optimizer_step=capture,
    )
    base = _base_model(model)
    gradient_error = _maximum_gradient_error(captured, expected_gradients)
    parameter_error = _maximum_parameter_error(base, reference)
    passed = _gradients_close(
        captured, expected_gradients, atol=args.atol, rtol=args.rtol
    ) and _parameters_close(base, reference, atol=args.atol, rtol=args.rtol)
    candidate = False
    if args.strategy == "bucket_async":
        launches = [
            event["timestamp_ns"]
            for event in result.timeline
            if event["kind"] == "collective_launch"
        ]
        backward_ends = [
            event["timestamp_ns"]
            for event in result.timeline
            if event["kind"] == "backward_complete"
        ]
        candidate = bool(launches and backward_ends and min(launches) < max(backward_ends))
    return {
        "rank": rank,
        "world_size": world_size,
        "strategy": args.strategy,
        "accumulation_steps": args.accumulation_steps,
        "gradient_max_error": gradient_error,
        "parameter_max_error": parameter_error,
        "none_gradient_names": sorted(
            name for name, gradient in captured.items() if gradient is None
        ),
        "correctness_passed": passed,
        "collective_count": result.collective_count,
        "payload_bytes": result.payload_bytes,
        "overlap_candidate": candidate,
        "timeline": result.timeline,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def run_amp_overflow_probe(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        raise ValueError("AMP overflow probe requires CUDA")
    config = model_preset(args.model_preset)
    tokens, labels = make_classification_batch(
        args.per_rank_batch_size,
        config,
        args.seed + 700 + dist.get_rank(),
    )
    tokens, labels = tokens.to(device), labels.to(device)
    model, reducer = _model_and_reducer(args, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=True)

    initial = [parameter.detach().clone() for parameter in model.parameters()]
    clean = train_step(
        model,
        optimizer,
        tokens,
        labels,
        reducer=reducer,
        precision="amp",
        scaler=scaler,
    )
    before_overflow = [parameter.detach().clone() for parameter in model.parameters()]
    clean_update = max(
        float((before - after).abs().max())
        for before, after in zip(initial, before_overflow, strict=True)
    )
    scale_before = float(scaler.get_scale())

    def inject_nonfinite_gradient(base: nn.Module) -> None:
        parameter = next(item for item in base.parameters() if item.grad is not None)
        assert parameter.grad is not None
        parameter.grad.view(-1)[0] = float("inf")

    overflow = train_step(
        model,
        optimizer,
        tokens,
        labels,
        reducer=reducer,
        precision="amp",
        scaler=scaler,
        before_unscale=inject_nonfinite_gradient,
    )
    after_overflow = [parameter.detach() for parameter in model.parameters()]
    parameter_max_error = max(
        float((before - after).abs().max())
        for before, after in zip(before_overflow, after_overflow, strict=True)
    )
    scale_after = float(scaler.get_scale())
    passed = (
        clean_update > 0
        and not clean.optimizer_step_skipped
        and overflow.optimizer_step_skipped
        and scale_after < scale_before
        and parameter_max_error == 0.0
    )
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "strategy": args.strategy,
        "case": "amp_overflow_probe",
        "clean_update_max_abs": clean_update,
        "clean_step_skipped": clean.optimizer_step_skipped,
        "overflow_step_skipped": overflow.optimizer_step_skipped,
        "scale_before_overflow": scale_before,
        "scale_after_overflow": scale_after,
        "parameter_max_error_after_skipped_step": parameter_max_error,
        "correctness_passed": passed,
    }


def run_benchmark(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    correctness = run_correctness(args, device)
    config = model_preset(args.model_preset)
    local_batch = args.per_rank_batch_size
    if local_batch % args.accumulation_steps:
        raise ValueError("per-rank batch must be divisible by accumulation steps")
    tokens, labels = make_classification_batch(
        local_batch, config, args.seed + 100 + dist.get_rank()
    )
    tokens, labels = tokens.to(device), labels.to(device)
    model, reducer = _model_and_reducer(args, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=args.precision == "amp")
    for _ in range(args.warmup_steps):
        train_step(
            model,
            optimizer,
            tokens,
            labels,
            reducer=reducer,
            accumulation_steps=args.accumulation_steps,
            precision=args.precision,
            scaler=scaler,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    local_times = []
    last_result = None
    dist.barrier()
    for _ in range(args.measured_steps):
        started = time.perf_counter()
        last_result = train_step(
            model,
            optimizer,
            tokens,
            labels,
            reducer=reducer,
            accumulation_steps=args.accumulation_steps,
            precision=args.precision,
            scaler=scaler,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        local_times.append(time.perf_counter() - started)
    gathered: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_times)
    slowest = [
        max(rank_times[index] for rank_times in gathered if rank_times is not None)
        for index in range(args.measured_steps)
    ]
    assert last_result is not None
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "strategy": args.strategy,
        "model_preset": args.model_preset,
        "precision": args.precision,
        "accumulation_steps": args.accumulation_steps,
        "bucket_cap_bytes": args.bucket_cap_bytes,
        "step_time_p50_ms": statistics.median(slowest) * 1000,
        "step_time_p95_ms": _percentile(slowest, 0.95) * 1000,
        "global_samples_per_second": local_batch
        * dist.get_world_size()
        / statistics.median(slowest),
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "collective_count_per_step": last_result.collective_count,
        "payload_bytes_per_step": last_result.payload_bytes,
        "timeline": last_result.timeline,
        "correctness_passed": correctness["correctness_passed"],
        "gradient_max_error": correctness["gradient_max_error"],
        "parameter_max_error": correctness["parameter_max_error"],
    }


def run_profile(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        raise ValueError("formal overlap profiling requires CUDA")
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    correctness = run_correctness(args, device)
    config = model_preset(args.model_preset)
    tokens, labels = make_classification_batch(
        args.per_rank_batch_size, config, args.seed + 500 + dist.get_rank()
    )
    tokens, labels = tokens.to(device), labels.to(device)
    model, reducer = _model_and_reducer(args, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    for _ in range(args.warmup_steps):
        train_step(model, optimizer, tokens, labels, reducer=reducer)
    dist.barrier()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        for _ in range(args.measured_steps):
            train_step(model, optimizer, tokens, labels, reducer=reducer)
    torch.cuda.synchronize(device)
    trace_path = args.rank_directory / f"rank_{dist.get_rank()}_trace.json"
    profiler.export_chrome_trace(str(trace_path))
    distributed_rows = []
    for row in profiler.key_averages():
        lowered = row.key.lower()
        if any(token in lowered for token in ("nccl", "allreduce", "c10d")):
            distributed_rows.append(
                {
                    "name": row.key,
                    "count": row.count,
                    "cpu_time_total_us": row.cpu_time_total,
                    "cuda_time_total_us": getattr(row, "device_time_total", 0.0),
                }
            )
    distributed_rows.sort(key=lambda row: row["cuda_time_total_us"], reverse=True)
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "strategy": args.strategy,
        "correctness_passed": correctness["correctness_passed"],
        "trace": str(trace_path),
        "distributed_rows": distributed_rows[:30],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("correctness", "benchmark", "profile", "amp_overflow_probe"),
        required=True,
    )
    parser.add_argument(
        "--strategy",
        choices=("bulk", "per_parameter", "bucket_sync", "bucket_async", "ddp"),
        required=True,
    )
    parser.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--rank-directory", type=Path, required=True)
    parser.add_argument("--model-preset", choices=("small", "medium"), default="small")
    parser.add_argument("--include-unused", action="store_true")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--per-rank-batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--bucket-cap-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--timeout-seconds", type=int, default=180)
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
    dist.init_process_group(args.backend, timeout=timedelta(seconds=args.timeout_seconds))
    try:
        function = {
            "correctness": run_correctness,
            "benchmark": run_benchmark,
            "profile": run_profile,
            "amp_overflow_probe": run_amp_overflow_probe,
        }[args.mode]
        payload = function(args, device)
        payload["status"] = "success"
        args.rank_directory.mkdir(parents=True, exist_ok=True)
        path = args.rank_directory / f"rank_{dist.get_rank()}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
