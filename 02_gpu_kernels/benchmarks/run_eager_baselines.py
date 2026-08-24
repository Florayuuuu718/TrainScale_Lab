"""Run CUDA eager/library baselines even when custom Triton launch is unavailable."""

from __future__ import annotations

import json
import math
import platform
import statistics
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

TensorFn = Callable[[], torch.Tensor]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def benchmark(name: str, function: TensorFn, *, inner: int, samples: int = 21) -> dict[str, Any]:
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    measurements: list[float] = []
    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            function()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end) * 1000.0 / inner)
    return {
        "name": name,
        "status": "success",
        "inner_iterations": inner,
        "samples": samples,
        "latency_us": {
            "median": statistics.median(measurements),
            "p10": percentile(measurements, 0.1),
            "p90": percentile(measurements, 0.9),
        },
        "peak_memory_delta_bytes": max(
            0, torch.cuda.max_memory_allocated() - baseline_memory
        ),
    }


def add_rate(
    case: dict[str, Any],
    *,
    bytes_moved: int | None = None,
    flops: int | None = None,
) -> None:
    seconds = case["latency_us"]["median"] * 1e-6
    if bytes_moved is not None:
        case["effective_bytes"] = bytes_moved
        case["effective_gb_per_s"] = bytes_moved / seconds / 1e9
    if flops is not None:
        case["flops"] = flops
        case["tflops"] = flops / seconds / 1e12


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def driver_version() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(20260823)
    device = torch.device("cuda")
    cases: list[dict[str, Any]] = []

    for size, inner in [(257, 2000), (65_536, 1000), (1_048_576, 100)]:
        x = torch.randn(size, device=device)
        y = torch.randn_like(x)

        def vector_function(x: torch.Tensor = x, y: torch.Tensor = y) -> torch.Tensor:
            return x + y

        case = benchmark(f"vector_add_n{size}", vector_function, inner=inner)
        case.update({"operator": "vector_add", "shape": [size], "dtype": "float32"})
        add_rate(case, bytes_moved=size * 4 * 3)
        cases.append(case)

    for size, inner in [(257, 2000), (65_536, 1000), (1_048_576, 100)]:
        x = torch.randn(size, device=device)
        bias = torch.randn_like(x)

        def relu_function(
            x: torch.Tensor = x, bias: torch.Tensor = bias
        ) -> torch.Tensor:
            return torch.relu(x + bias)

        separate = benchmark(
            f"relu_add_separate_n{size}", relu_function, inner=inner
        )
        separate.update(
            {"operator": "relu_add", "implementation": "pytorch_eager", "shape": [size]}
        )
        cases.append(separate)

    for rows, cols, inner in [(1, 17, 2000), (32, 127, 1000), (256, 1024, 200)]:
        x = torch.randn((rows, cols), device=device) * 20

        def softmax_function(x: torch.Tensor = x) -> torch.Tensor:
            return torch.softmax(x, dim=-1)

        case = benchmark(
            f"softmax_{rows}x{cols}",
            softmax_function,
            inner=inner,
        )
        case.update({"operator": "softmax", "shape": [rows, cols], "dtype": "float32"})
        add_rate(case, bytes_moved=rows * cols * 4 * 2)
        cases.append(case)

    for rows, hidden, inner in [(1, 17, 2000), (32, 127, 1000), (256, 1024, 200)]:
        x = torch.randn((rows, hidden), device=device)
        weight = torch.randn(hidden, device=device)
        bias = torch.randn(hidden, device=device)

        def layer_norm_function(
            x: torch.Tensor = x,
            weight: torch.Tensor = weight,
            bias: torch.Tensor = bias,
            hidden: int = hidden,
        ) -> torch.Tensor:
            return functional.layer_norm(x, (hidden,), weight, bias)

        case = benchmark(
            f"layer_norm_{rows}x{hidden}",
            layer_norm_function,
            inner=inner,
        )
        case.update({"operator": "layer_norm", "shape": [rows, hidden], "dtype": "float32"})
        cases.append(case)

    for m_size, n_size, k_size, inner in [
        (17, 31, 23, 1000),
        (128, 128, 128, 500),
        (512, 512, 512, 100),
        (509, 509, 509, 100),
    ]:
        a = torch.randn((m_size, k_size), device=device, dtype=torch.float16)
        b = torch.randn((k_size, n_size), device=device, dtype=torch.float16)

        def matmul_function(a: torch.Tensor = a, b: torch.Tensor = b) -> torch.Tensor:
            return a @ b

        case = benchmark(
            f"matmul_{m_size}x{n_size}x{k_size}",
            matmul_function,
            inner=inner,
        )
        case.update(
            {
                "operator": "matmul",
                "shape": [m_size, n_size, k_size],
                "dtype": "float16",
            }
        )
        add_rate(case, flops=2 * m_size * n_size * k_size)
        cases.append(case)

    attention_errors: list[dict[str, Any]] = []
    for heads, sequence, head_dim, causal, inner in [
        (2, 64, 32, False, 200),
        (8, 128, 64, False, 100),
        (8, 128, 64, True, 100),
        (8, 257, 64, False, 50),
    ]:
        query = torch.randn(
            (1, heads, sequence, head_dim), device=device, dtype=torch.float16
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        def sdpa(
            query: torch.Tensor = query,
            key: torch.Tensor = key,
            value: torch.Tensor = value,
            causal: bool = causal,
        ) -> torch.Tensor:
            return functional.scaled_dot_product_attention(
                query, key, value, is_causal=causal
            )
        causal_mask = (
            torch.ones((sequence, sequence), device=device, dtype=torch.bool).tril()
            if causal
            else None
        )

        def explicit_attention(
            query: torch.Tensor = query,
            key: torch.Tensor = key,
            value: torch.Tensor = value,
            causal_mask: torch.Tensor | None = causal_mask,
            scale: float = 1.0 / math.sqrt(head_dim),
        ) -> torch.Tensor:
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            if causal_mask is not None:
                scores = scores.masked_fill(~causal_mask, -float("inf"))
            return torch.matmul(torch.softmax(scores, dim=-1), value)

        explicit_case = benchmark(
            f"explicit_attention_h{heads}_s{sequence}_d{head_dim}_causal{causal}",
            explicit_attention,
            inner=inner,
        )
        explicit_case.update(
            {
                "operator": "attention",
                "implementation": "pytorch_explicit",
                "shape": [1, heads, sequence, head_dim],
                "dtype": "float16",
                "causal": causal,
            }
        )
        add_rate(explicit_case, flops=4 * heads * sequence * sequence * head_dim)
        cases.append(explicit_case)

        case = benchmark(
            f"sdpa_h{heads}_s{sequence}_d{head_dim}_causal{causal}",
            sdpa,
            inner=inner,
        )
        case.update(
            {
                "operator": "attention",
                "implementation": "pytorch_sdpa",
                "shape": [1, heads, sequence, head_dim],
                "dtype": "float16",
                "causal": causal,
            }
        )
        add_rate(case, flops=4 * heads * sequence * sequence * head_dim)
        cases.append(case)

        explicit = explicit_attention()
        actual = sdpa()
        attention_errors.append(
            {
                "shape": [1, heads, sequence, head_dim],
                "causal": causal,
                "max_abs_error": (actual.float() - explicit.float()).abs().max().item(),
            }
        )

    properties = torch.cuda.get_device_properties(0)
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability()),
            "total_memory_bytes": properties.total_memory,
            "driver": driver_version(),
        },
        "measurement": {
            "timer": "torch.cuda.Event",
            "warmup_calls": 10,
            "samples": 21,
            "note": "Allocation and framework dispatch are included in logical operator latency.",
        },
        "triton_status": {
            "status": "blocked",
            "symptom": "SIGSEGV while Triton CompiledKernel initializes launch handles",
            "minimal_torch_compile_exit_code": 1,
            "vector_add_single_warp_exit_code": 1,
            "softmax_single_load_exit_code": 1,
        },
        "attention_sdpa_vs_explicit": attention_errors,
        "cases": cases,
    }
    output = Path("02_gpu_kernels/results/eager_baselines.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(f"cases={len(cases)}")


if __name__ == "__main__":
    main()
