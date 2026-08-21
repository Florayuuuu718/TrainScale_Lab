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
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        acc_events=True,
    ) as profiler:
        model.train()
        for step, (features, targets) in enumerate(data.train_loader):
            if step >= 5:
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
    events = sorted(profiler.key_averages(), key=lambda event: event.cpu_time_total, reverse=True)
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
            "events": rows,
        },
    )


if __name__ == "__main__":
    main()
