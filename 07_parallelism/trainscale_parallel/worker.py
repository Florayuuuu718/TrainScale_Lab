"""Torchrun worker for local TP correctness and GPU FSDP2/TP experiments."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn

from trainscale_parallel.tensor_parallel import (
    HeadParallelSelfAttention,
    ReferenceMLP,
    ReferenceSelfAttention,
    TensorParallelMLP,
)


def _maximum_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max())


def _parameter_comparisons_mlp(
    parallel: TensorParallelMLP, reference: ReferenceMLP
) -> dict[str, float]:
    hidden_slice = parallel.hidden_slice
    return {
        "fc1_weight": _maximum_error(parallel.fc1_weight, reference.fc1.weight[hidden_slice]),
        "fc1_bias": _maximum_error(parallel.fc1_bias, reference.fc1.bias[hidden_slice]),
        "fc2_weight": _maximum_error(parallel.fc2_weight, reference.fc2.weight[:, hidden_slice]),
        "output_bias": _maximum_error(parallel.output_bias, reference.fc2.bias),
    }


def _gradient_comparisons_mlp(
    parallel: TensorParallelMLP, reference: ReferenceMLP
) -> dict[str, float]:
    hidden_slice = parallel.hidden_slice
    assert parallel.fc1_weight.grad is not None and reference.fc1.weight.grad is not None
    assert parallel.fc1_bias.grad is not None and reference.fc1.bias.grad is not None
    assert parallel.fc2_weight.grad is not None and reference.fc2.weight.grad is not None
    assert parallel.output_bias.grad is not None and reference.fc2.bias.grad is not None
    return {
        "fc1_weight": _maximum_error(
            parallel.fc1_weight.grad, reference.fc1.weight.grad[hidden_slice]
        ),
        "fc1_bias": _maximum_error(parallel.fc1_bias.grad, reference.fc1.bias.grad[hidden_slice]),
        "fc2_weight": _maximum_error(
            parallel.fc2_weight.grad, reference.fc2.weight.grad[:, hidden_slice]
        ),
        "output_bias": _maximum_error(parallel.output_bias.grad, reference.fc2.bias.grad),
    }


def run_tp_mlp(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    reference = ReferenceMLP(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    parallel = TensorParallelMLP(reference).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    inputs = torch.randn(args.batch_size, args.input_dim, generator=generator, device=device)
    targets = torch.randn(args.batch_size, args.output_dim, generator=generator, device=device)
    reference_output = reference(inputs)
    parallel_output = parallel(inputs)
    output_error = _maximum_error(parallel_output, reference_output)
    reference_loss = nn.functional.mse_loss(reference_output, targets)
    parallel_loss = nn.functional.mse_loss(parallel_output, targets) / dist.get_world_size()
    reference_loss.backward()
    parallel_loss.backward()
    parallel.synchronize_replicated_gradients()
    gradient_errors = _gradient_comparisons_mlp(parallel, reference)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=args.learning_rate)
    parallel_optimizer = torch.optim.SGD(parallel.parameters(), lr=args.learning_rate)
    reference_optimizer.step()
    parallel_optimizer.step()
    parameter_errors = _parameter_comparisons_mlp(parallel, reference)
    maximum = max(output_error, *gradient_errors.values(), *parameter_errors.values())
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "case": "tp_mlp",
        "output_max_error": output_error,
        "gradient_errors": gradient_errors,
        "parameter_update_errors": parameter_errors,
        "maximum_error": maximum,
        "correctness_passed": maximum <= args.atol,
        "local_shapes": parallel.local_shapes(),
        "placements": {
            "fc1_weight": "Shard(output_dim)",
            "fc2_weight": "Shard(input_dim)",
            "output": "Replicate",
        },
        "collectives_per_forward": 1,
    }


def _attention_parameter_errors(
    parallel: HeadParallelSelfAttention, reference: ReferenceSelfAttention
) -> dict[str, float]:
    width_slice = parallel.width_slice
    errors = {}
    for name in ("query", "key", "value"):
        errors[f"{name}_weight"] = _maximum_error(
            getattr(parallel, f"{name}_weight"),
            getattr(reference, name).weight[width_slice],
        )
        errors[f"{name}_bias"] = _maximum_error(
            getattr(parallel, f"{name}_bias"),
            getattr(reference, name).bias[width_slice],
        )
    errors["output_weight"] = _maximum_error(
        parallel.output_weight, reference.output.weight[:, width_slice]
    )
    errors["output_bias"] = _maximum_error(parallel.output_bias, reference.output.bias)
    return errors


def _attention_gradient_errors(
    parallel: HeadParallelSelfAttention, reference: ReferenceSelfAttention
) -> dict[str, float]:
    width_slice = parallel.width_slice
    errors = {}
    for name in ("query", "key", "value"):
        local_weight = getattr(parallel, f"{name}_weight").grad
        local_bias = getattr(parallel, f"{name}_bias").grad
        full_weight = getattr(reference, name).weight.grad
        full_bias = getattr(reference, name).bias.grad
        assert all(item is not None for item in (local_weight, local_bias, full_weight, full_bias))
        errors[f"{name}_weight"] = _maximum_error(local_weight, full_weight[width_slice])
        errors[f"{name}_bias"] = _maximum_error(local_bias, full_bias[width_slice])
    assert parallel.output_weight.grad is not None
    assert parallel.output_bias.grad is not None
    assert reference.output.weight.grad is not None
    assert reference.output.bias.grad is not None
    errors["output_weight"] = _maximum_error(
        parallel.output_weight.grad, reference.output.weight.grad[:, width_slice]
    )
    errors["output_bias"] = _maximum_error(parallel.output_bias.grad, reference.output.bias.grad)
    return errors


def run_tp_attention(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    reference = ReferenceSelfAttention(args.d_model, args.num_heads).to(device)
    parallel = HeadParallelSelfAttention(reference).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    inputs = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.d_model,
        generator=generator,
        device=device,
    )
    targets = torch.randn(inputs.shape, generator=generator, device=device)
    reference_output = reference(inputs)
    parallel_output = parallel(inputs)
    output_error = _maximum_error(parallel_output, reference_output)
    reference_loss = nn.functional.mse_loss(reference_output, targets)
    parallel_loss = nn.functional.mse_loss(parallel_output, targets) / dist.get_world_size()
    reference_loss.backward()
    parallel_loss.backward()
    parallel.synchronize_replicated_gradients()
    gradient_errors = _attention_gradient_errors(parallel, reference)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=args.learning_rate)
    parallel_optimizer = torch.optim.SGD(parallel.parameters(), lr=args.learning_rate)
    reference_optimizer.step()
    parallel_optimizer.step()
    parameter_errors = _attention_parameter_errors(parallel, reference)
    maximum = max(output_error, *gradient_errors.values(), *parameter_errors.values())
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "case": "tp_attention",
        "output_max_error": output_error,
        "gradient_errors": gradient_errors,
        "parameter_update_errors": parameter_errors,
        "maximum_error": maximum,
        "correctness_passed": maximum <= args.atol,
        "local_shapes": parallel.local_shapes(),
        "placements": {
            "qkv_heads": "Shard(heads)",
            "output_weight": "Shard(input_dim)",
            "output": "Replicate",
        },
        "collectives_per_forward": 1,
    }


def run_fsdp2_probe(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    import torch.distributed.checkpoint as dcp  # noqa: PLC0415
    from torch.distributed.checkpoint.state_dict import (  # noqa: PLC0415
        get_state_dict,
        set_state_dict,
    )
    from torch.distributed.device_mesh import init_device_mesh  # noqa: PLC0415
    from torch.distributed.fsdp import fully_shard  # noqa: PLC0415
    from torch.distributed.tensor import DTensor  # noqa: PLC0415
    from trainscale_engine.model import (  # noqa: PLC0415
        TinyTransformer,
        make_classification_batch,
        model_preset,
    )

    torch.manual_seed(args.seed)
    model_config = model_preset("small")
    reference = TinyTransformer(model_config).to(device)
    torch.manual_seed(args.seed)
    model = TinyTransformer(model_config).to(device)
    mesh = init_device_mesh(device.type, (dist.get_world_size(),), mesh_dim_names=("fsdp",))
    fully_shard(model, mesh=mesh)
    sharded_before = all(isinstance(parameter, DTensor) for parameter in model.parameters())
    local_shapes = [
        tuple(cast(Any, parameter).to_local().shape) for parameter in model.parameters()
    ]
    reference_optimizer = torch.optim.SGD(
        reference.parameters(), lr=args.learning_rate, momentum=0.9
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    global_inputs, global_targets = make_classification_batch(
        args.batch_size * dist.get_world_size(), model_config, args.seed + 9
    )
    global_inputs, global_targets = global_inputs.to(device), global_targets.to(device)
    start = dist.get_rank() * args.batch_size
    stop = start + args.batch_size

    def reference_step() -> None:
        reference_optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(reference(global_inputs), global_targets)
        loss.backward()
        reference_optimizer.step()

    def sharded_step(module: nn.Module, optim: torch.optim.Optimizer) -> None:
        optim.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(
            module(global_inputs[start:stop]), global_targets[start:stop]
        )
        loss.backward()
        optim.step()

    def full_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: value.full_tensor() if isinstance(value, DTensor) else value.detach()
            for name, value in module.state_dict().items()
        }

    reference_step()
    sharded_step(model, optimizer)
    first_state = full_state_dict(model)
    first_step_errors = {
        name: _maximum_error(first_state[name], expected)
        for name, expected in reference.state_dict().items()
    }
    model_state, optimizer_state = get_state_dict(model, optimizer)
    checkpoint = args.rank_directory / "dcp"
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=checkpoint,
    )
    sharded_step(model, optimizer)
    continuous_state = full_state_dict(model)

    torch.manual_seed(args.seed)
    restored = TinyTransformer(model_config).to(device)
    fully_shard(restored, mesh=mesh)
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=args.learning_rate, momentum=0.9)
    restored_model_state, restored_optimizer_state = get_state_dict(restored, restored_optimizer)
    restored_payload = {
        "model": restored_model_state,
        "optimizer": restored_optimizer_state,
    }
    dcp.load(restored_payload, checkpoint_id=checkpoint)
    set_state_dict(
        restored,
        restored_optimizer,
        model_state_dict=restored_model_state,
        optim_state_dict=restored_optimizer_state,
    )
    sharded_step(restored, restored_optimizer)
    restored_state = full_state_dict(restored)
    resume_errors = {
        name: _maximum_error(restored_state[name], expected)
        for name, expected in continuous_state.items()
    }
    maximum = max(*first_step_errors.values(), *resume_errors.values(), 0.0)
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "case": "fsdp2_probe",
        "sharded_parameters_before_forward": sharded_before,
        "local_parameter_shapes": local_shapes,
        "parameter_update_errors": first_step_errors,
        "checkpoint_resume_errors": resume_errors,
        "checkpoint_files": sorted(path.name for path in checkpoint.iterdir()),
        "maximum_error": maximum,
        "correctness_passed": sharded_before and maximum <= args.atol,
        "placements": [
            [str(placement) for placement in parameter.placements]
            for parameter in model.parameters()
            if isinstance(parameter, DTensor)
        ],
    }


def run_native_tp_probe(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from torch.distributed.device_mesh import init_device_mesh  # noqa: PLC0415
    from torch.distributed.tensor import DTensor  # noqa: PLC0415
    from torch.distributed.tensor.parallel import (  # noqa: PLC0415
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    torch.manual_seed(args.seed)
    reference = ReferenceMLP(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    torch.manual_seed(args.seed)
    model = ReferenceMLP(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    mesh = init_device_mesh(device.type, (dist.get_world_size(),), mesh_dim_names=("tp",))
    parallelize_module(
        model,
        mesh,
        {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()},
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 17)
    inputs = torch.randn(args.batch_size, args.input_dim, generator=generator, device=device)
    targets = torch.randn(args.batch_size, args.output_dim, generator=generator, device=device)
    reference_output = reference(inputs)
    parallel_output = model(inputs)
    output_error = _maximum_error(parallel_output, reference_output)
    reference_loss = nn.functional.mse_loss(reference_output, targets)
    parallel_loss = nn.functional.mse_loss(parallel_output, targets)
    reference_loss.backward()
    parallel_loss.backward()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    reference_optimizer.step()
    optimizer.step()
    full_state = {
        name: value.full_tensor() if isinstance(value, DTensor) else value.detach()
        for name, value in model.state_dict().items()
    }
    errors = {
        name: _maximum_error(full_state[name], expected)
        for name, expected in reference.state_dict().items()
    }
    maximum = max(output_error, *errors.values())
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "case": "native_tp_probe",
        "output_max_error": output_error,
        "parameter_update_errors": errors,
        "maximum_error": maximum,
        "correctness_passed": maximum <= args.atol,
        "local_shapes": {
            name: tuple(value.to_local().shape)
            for name, value in model.state_dict().items()
            if isinstance(value, DTensor)
        },
        "placements": {
            name: [str(placement) for placement in value.placements]
            for name, value in model.state_dict().items()
            if isinstance(value, DTensor)
        },
    }


def _nearest_percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def run_benchmark(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from torch.distributed.device_mesh import init_device_mesh  # noqa: PLC0415
    from torch.distributed.fsdp import fully_shard  # noqa: PLC0415
    from torch.distributed.tensor import DTensor  # noqa: PLC0415
    from torch.distributed.tensor.parallel import (  # noqa: PLC0415
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )
    from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: PLC0415
    from trainscale_engine.model import (  # noqa: PLC0415
        TinyTransformer,
        make_classification_batch,
        model_preset,
    )

    torch.manual_seed(args.seed)
    model_config = model_preset(args.model_preset)
    family: str
    if args.strategy in {"ddp", "fsdp_root", "fsdp_layer"}:
        base = TinyTransformer(model_config).to(device)
        if args.strategy == "ddp":
            local_rank = int(os.environ["LOCAL_RANK"])
            model: nn.Module = DDP(base, device_ids=[local_rank], output_device=local_rank)
        else:
            mesh = init_device_mesh(device.type, (dist.get_world_size(),), mesh_dim_names=("fsdp",))
            if args.strategy == "fsdp_layer":
                for layer in base.encoder.layers:
                    fully_shard(layer, mesh=mesh)
            fully_shard(base, mesh=mesh)
            model = base
        tokens, labels = make_classification_batch(
            args.per_rank_batch_size,
            model_config,
            args.seed + 100 + dist.get_rank(),
        )
        tokens, labels = tokens.to(device), labels.to(device)
        family = "data_parallel_transformer"

        def one_step() -> None:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(tokens), labels)
            loss.backward()
            optimizer.step()

        effective_global_batch = args.per_rank_batch_size * dist.get_world_size()
    else:
        reference_mlp = ReferenceMLP(
            model_config.d_model, model_config.feedforward_dim, model_config.d_model
        ).to(device)
        if args.strategy == "tp":
            mesh = init_device_mesh(device.type, (dist.get_world_size(),), mesh_dim_names=("tp",))
            parallelize_module(
                reference_mlp,
                mesh,
                {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()},
            )
        model = reference_mlp
        generator = torch.Generator(device=device).manual_seed(args.seed + 200)
        features = torch.randn(
            args.per_rank_batch_size,
            model_config.d_model,
            generator=generator,
            device=device,
        )
        targets = torch.randn(features.shape, generator=generator, device=device)
        family = "tensor_parallel_mlp"

        def one_step() -> None:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(features), targets)
            loss.backward()
            optimizer.step()

        effective_global_batch = args.per_rank_batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for _ in range(args.warmup_steps):
        one_step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    local_times = []

    def measured_step() -> None:
        started = time.perf_counter()
        one_step()
        torch.cuda.synchronize(device)
        local_times.append(time.perf_counter() - started)

    trace_path = None
    if args.trace_directory is not None:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            for _ in range(args.measured_steps):
                measured_step()
                profiler.step()
        args.trace_directory.mkdir(parents=True, exist_ok=True)
        trace_path = args.trace_directory / f"rank_{dist.get_rank()}.json"
        profiler.export_chrome_trace(str(trace_path))
    else:
        for _ in range(args.measured_steps):
            measured_step()
    gathered: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_times)
    slowest = [
        max(rank_times[index] for rank_times in gathered if rank_times is not None)
        for index in range(args.measured_steps)
    ]
    p50 = statistics.median(slowest)
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "case": "gpu_benchmark",
        "strategy": args.strategy,
        "family": family,
        "model_preset": args.model_preset,
        "step_time_p50_ms": p50 * 1000,
        "step_time_p95_ms": _nearest_percentile(slowest, 0.95) * 1000,
        "samples_per_second": effective_global_batch / p50,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "local_parameter_bytes": sum(
            (parameter.to_local().numel() if isinstance(parameter, DTensor) else parameter.numel())
            * parameter.element_size()
            for parameter in model.parameters()
        ),
        "correctness_source": "separate_preflight_probe",
        "trace": str(trace_path.resolve()) if trace_path is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "tp_mlp",
            "tp_attention",
            "fsdp2_probe",
            "native_tp_probe",
            "benchmark",
        ),
        required=True,
    )
    parser.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--rank-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--strategy",
        choices=("ddp", "fsdp_root", "fsdp_layer", "tp", "tp_reference"),
        default="ddp",
    )
    parser.add_argument("--model-preset", choices=("small", "medium"), default="medium")
    parser.add_argument("--per-rank-batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=10)
    parser.add_argument("--trace-directory", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--timeout-seconds", type=int, default=120)
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
        payload = {
            "tp_mlp": run_tp_mlp,
            "tp_attention": run_tp_attention,
            "fsdp2_probe": run_fsdp2_probe,
            "native_tp_probe": run_native_tp_probe,
            "benchmark": run_benchmark,
        }[args.mode](args, device)
        payload["status"] = "success"
        args.rank_directory.mkdir(parents=True, exist_ok=True)
        (args.rank_directory / f"rank_{dist.get_rank()}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
