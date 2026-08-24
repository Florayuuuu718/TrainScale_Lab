"""Profile representative memory- and compute-oriented PyTorch CUDA baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def profile_case(name: str, function: object) -> dict[str, Any]:
    callable_function = function
    for _ in range(10):
        callable_function()  # type: ignore[operator]
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for _ in range(20):
            callable_function()  # type: ignore[operator]
    torch.cuda.synchronize()
    events = list(profiler.key_averages())
    device_rows = [
        {
            "name": event.key,
            "count": event.count,
            "device_time_total_us": getattr(event, "device_time_total", 0.0),
            "cpu_time_total_us": event.cpu_time_total,
            "device_memory_usage_bytes": getattr(event, "device_memory_usage", 0),
        }
        for event in events
        if getattr(event, "device_time_total", 0.0) > 0
    ]
    device_rows.sort(key=lambda row: row["device_time_total_us"], reverse=True)
    trace = Path(f"02_gpu_kernels/results/raw/{name}_trace.json")
    trace.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace))
    return {
        "name": name,
        "iterations": 20,
        "device_rows": device_rows[:15],
        "trace": str(trace),
        "aggregation_note": (
            "Rows come from key_averages and can be nested; summed row time is not GPU wall time."
        ),
    }


def main() -> None:
    vector_x = torch.randn(1_048_576, device="cuda")
    vector_y = torch.randn_like(vector_x)
    matrix_a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    matrix_b = torch.randn_like(matrix_a)
    payload = {
        "vector_add": profile_case("vector_add_1m", lambda: vector_x + vector_y),
        "matmul": profile_case("matmul_512", lambda: matrix_a @ matrix_b),
    }
    output = Path("02_gpu_kernels/results/profiler_summary.json")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
