"""Export a short PyTorch Profiler trace and a compact operator summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from .config import load_config
from .data import make_data
from .engine import seed_everything
from .models import make_model
from .reporting import environment_record, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--wait-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--active-steps", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    seed_everything(config.seed)
    device = torch.device(config.device)
    data = make_data(config)
    model = make_model(config).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    supported_activities = torch.profiler.supported_activities()
    requested_cuda = device.type == "cuda"
    cuda_supported = torch.profiler.ProfilerActivity.CUDA in supported_activities
    activities = [torch.profiler.ProfilerActivity.CPU]
    if requested_cuda and cuda_supported:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=args.wait_steps,
            warmup=args.warmup_steps,
            active=args.active_steps,
            repeat=1,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        acc_events=True,
    ) as profiler:
        model.train()
        total_steps = args.wait_steps + args.warmup_steps + args.active_steps
        for step, (features, targets) in enumerate(data.train_loader):
            if step >= total_steps:
                break
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.profiler.record_function("train_step"):
                logits = model(features)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
            profiler.step()

    trace = Path(args.trace)
    trace.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace))
    rows = []
    events = list(profiler.key_averages())
    device_events = [
        event for event in events if getattr(event, "device_time_total", 0.0) > 0
    ]
    summed_device_time_us = sum(
        getattr(event, "device_time_total", 0.0) for event in events
    )
    events.sort(key=lambda event: event.cpu_time_total, reverse=True)
    for event in events[:30]:
        rows.append(
            {
                "name": event.key,
                "count": event.count,
                "cpu_time_total_us": event.cpu_time_total,
                "device_time_total_us": getattr(event, "device_time_total", 0.0),
                "cpu_memory_usage_bytes": event.cpu_memory_usage,
                "device_memory_usage_bytes": getattr(event, "device_memory_usage", 0),
            }
        )
    write_json(
        args.summary,
        {
            "config": config.to_dict(),
            "environment": environment_record(),
            "trace": str(trace),
            "profiler_status": {
                "requested_cuda": requested_cuda,
                "cuda_reported_supported": cuda_supported,
                "cuda_events_present": bool(device_events),
                "device_time_aggregate_row_count": len(device_events),
                "summed_device_time_across_aggregates_us": summed_device_time_us,
                "aggregation_note": (
                    "Counts and summed time come from profiler.key_averages(); nested "
                    "operator rows can overlap, so they are not raw kernel count or GPU wall time."
                ),
                # Backward-compatible aliases retained for earlier result files.
                "cuda_event_count": len(device_events),
                "total_device_time_us": summed_device_time_us,
                "wait_steps": args.wait_steps,
                "warmup_steps": args.warmup_steps,
                "active_steps": args.active_steps,
            },
            "events": rows,
        },
    )


if __name__ == "__main__":
    main()
