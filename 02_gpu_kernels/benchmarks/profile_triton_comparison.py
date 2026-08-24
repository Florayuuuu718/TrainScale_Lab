"""Profile representative PyTorch and Triton kernels after correctness passes."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "02_gpu_kernels"))

from trainscale_kernels import attention, matmul, relu_add, vector_add  # noqa: E402


def profile_case(
    name: str,
    function: Any,
    *,
    trace_directory: Path,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for _ in range(iterations):
            function()
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
    trace_directory.mkdir(parents=True, exist_ok=True)
    trace = trace_directory / f"{name}_trace.json"
    profiler.export_chrome_trace(str(trace))
    return {
        "name": name,
        "iterations": iterations,
        "device_rows": device_rows[:15],
        "trace": str(trace),
        "aggregation_note": (
            "Rows come from key_averages and can be nested; do not sum them as GPU wall time."
        ),
    }


def environment_manifest() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "triton_package": version("triton"),
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": properties.total_memory,
        "driver": driver,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("02_gpu_kernels/results/triton_profiler_summary.json"),
    )
    parser.add_argument(
        "--trace-directory",
        type=Path,
        default=Path("02_gpu_kernels/results/raw/triton_profiler"),
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--allow-mounted-path", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    repository_path = str(REPOSITORY_ROOT.resolve())
    performance_location_ok = repository_path.startswith("/home/")
    if not performance_location_ok and not args.allow_mounted_path:
        raise SystemExit(
            "Formal profiles must use a WSL /home/... checkout; "
            "pass --allow-mounted-path only for a non-reportable smoke run."
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(20260824)
    vector_x = torch.randn(1_048_576, device="cuda")
    vector_y = torch.randn_like(vector_x)
    addend = torch.randn_like(vector_x)
    matrix_left = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    matrix_right = torch.randn_like(matrix_left)
    query = torch.randn((8, 128, 64), device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    cases = {
        "vector_add_pytorch": lambda: vector_x + vector_y,
        "vector_add_triton": lambda: vector_add(vector_x, vector_y),
        "relu_add_pytorch": lambda: torch.relu(vector_x + addend),
        "relu_add_triton": lambda: relu_add(vector_x, addend),
        "matmul_pytorch": lambda: matrix_left @ matrix_right,
        "matmul_triton": lambda: matmul(matrix_left, matrix_right),
        "attention_pytorch_sdpa": lambda: torch.nn.functional.scaled_dot_product_attention(
            query[None], key[None], value[None]
        )[0],
        "attention_triton": lambda: attention(query, key, value),
    }
    summaries = {
        name: profile_case(
            name,
            function,
            trace_directory=args.trace_directory,
            iterations=args.iterations,
        )
        for name, function in cases.items()
    }
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "repository_root": repository_path,
        "performance_location_ok": performance_location_ok,
        "environment": environment_manifest(),
        "protocol": {
            "correctness_precondition": (
                "Run check_environment.py and test_triton_ops.py before this profiler."
            ),
            "warmup_calls": 10,
            "iterations": args.iterations,
            "scope": "representative forward cases",
        },
        "cases": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(f"cases={len(summaries)}")


if __name__ == "__main__":
    main()
