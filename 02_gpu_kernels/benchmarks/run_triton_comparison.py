"""Crash-isolated PyTorch/Triton forward benchmarks for module 02."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_contract import (  # noqa: E402
    load_case_ids,
    load_cases,
    percentile,
    select_case_ids,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULT_MARKER = "TRAINSCALE_BENCHMARK_RESULT="
SEED = 20260824

CONFIG_ROOT = REPOSITORY_ROOT / "02_gpu_kernels" / "configs"
CASES = load_cases(CONFIG_ROOT / "benchmark_full.toml")
SMOKE_CASES = load_case_ids(CONFIG_ROOT / "benchmark_smoke.toml", CASES)
FULL_CASES = tuple(CASES)


def shell_exit_code(returncode: int) -> int:
    return 128 - returncode if returncode < 0 else returncode


def tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[-limit:]


def dtype_from_name(torch: Any, name: str) -> Any:
    return {"float16": torch.float16, "float32": torch.float32}[name]


def build_callables(case: dict[str, Any], torch: Any) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(REPOSITORY_ROOT / "02_gpu_kernels"))
    from trainscale_kernels import (  # noqa: PLC0415
        attention,
        layer_norm,
        matmul,
        relu_add,
        softmax,
        vector_add,
    )

    torch.manual_seed(SEED)
    device = torch.device("cuda")
    dtype = dtype_from_name(torch, case["dtype"])
    operator = case["operator"]
    shape = case["shape"]

    if operator == "vector_add":
        x = torch.randn(shape, device=device, dtype=dtype)
        y = torch.randn_like(x)
        def reference() -> Any:
            return x + y

        return reference, lambda: vector_add(x, y), reference
    if operator == "relu_add":
        x = torch.randn(shape, device=device, dtype=dtype)
        addend = torch.randn_like(x)
        def reference() -> Any:
            return torch.relu(x + addend)

        return reference, lambda: relu_add(x, addend), reference
    if operator == "softmax":
        x = torch.randn(shape, device=device, dtype=dtype) * 20
        def reference() -> Any:
            return torch.softmax(x, dim=-1)

        return reference, lambda: softmax(x), reference
    if operator == "layer_norm":
        rows, hidden = shape
        x = torch.randn((rows, hidden), device=device, dtype=dtype)
        weight = torch.randn(hidden, device=device, dtype=dtype)
        bias = torch.randn(hidden, device=device, dtype=dtype)
        def reference() -> Any:
            return torch.native_layer_norm(x, (hidden,), weight, bias, 1e-5)

        def candidate() -> Any:
            return layer_norm(x, weight, bias)

        return reference, candidate, reference
    if operator == "matmul":
        rows, cols, reduction = shape
        left = torch.randn((rows, reduction), device=device, dtype=dtype)
        right = torch.randn((reduction, cols), device=device, dtype=dtype)
        def reference() -> Any:
            return left @ right

        def accuracy_reference() -> Any:
            return left.float() @ right.float()

        return reference, lambda: matmul(left, right), accuracy_reference
    if operator == "attention":
        heads, sequence, head_dim = shape
        query = torch.randn((heads, sequence, head_dim), device=device, dtype=dtype)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        causal = case["causal"]
        def reference() -> Any:
            return torch.nn.functional.scaled_dot_product_attention(
                query[None], key[None], value[None], is_causal=causal
            )[0]

        def candidate() -> Any:
            return attention(query, key, value, causal=causal)

        return reference, candidate, reference
    raise ValueError(f"Unknown operator: {operator}")


def normalized_outputs(operator: str, output: Any) -> tuple[Any, ...]:
    if operator != "layer_norm":
        return (output,)
    values, mean, rstd = output
    return values, mean.reshape(-1), rstd.reshape(-1)


def correctness_report(
    operator: str,
    actual: Any,
    expected: Any,
    *,
    atol: float,
    rtol: float,
    torch: Any,
) -> dict[str, Any]:
    actual_values = normalized_outputs(operator, actual)
    expected_values = normalized_outputs(operator, expected)
    max_absolute_error = 0.0
    max_relative_error = 0.0
    try:
        for actual_tensor, expected_tensor in zip(actual_values, expected_values, strict=True):
            difference = (actual_tensor.float() - expected_tensor.float()).abs()
            max_absolute_error = max(max_absolute_error, difference.max().item())
            relative = difference / expected_tensor.float().abs().clamp_min(1e-12)
            max_relative_error = max(max_relative_error, relative.max().item())
            torch.testing.assert_close(
                actual_tensor,
                expected_tensor,
                atol=atol,
                rtol=rtol,
                check_dtype=False,
            )
    except AssertionError as error:
        return {
            "status": "failed",
            "atol": atol,
            "rtol": rtol,
            "max_absolute_error": max_absolute_error,
            "max_relative_error": max_relative_error,
            "error": tail(str(error)),
        }
    return {
        "status": "passed",
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
    }


def tolerances(case: dict[str, Any]) -> tuple[float, float]:
    operator = case["operator"]
    if operator == "softmax":
        return 2e-6, 2e-5
    if operator == "layer_norm":
        return 5e-5, 5e-5
    if operator == "matmul":
        reduction = case["shape"][2]
        return (2.5e-2 if reduction >= 256 else 2e-2), 2e-2
    if operator == "attention":
        return 3e-2, 3e-2
    return 1e-5, 1e-5


def steady_benchmark(
    function: Any,
    *,
    inner: int,
    samples: int,
    warmup: int,
    torch: Any,
) -> dict[str, Any]:
    last_output: Any = None
    for _ in range(warmup):
        last_output = function()
    torch.cuda.synchronize()
    measurements: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            last_output = function()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end) * 1000.0 / inner)
    del last_output
    return {
        "timer": "torch.cuda.Event",
        "inner_iterations": inner,
        "samples": samples,
        "warmup_calls": warmup,
        "latency_us": {
            "median": statistics.median(measurements),
            "p10": percentile(measurements, 0.1),
            "p90": percentile(measurements, 0.9),
        },
    }


def rate_metrics(case: dict[str, Any], median_us: float) -> dict[str, float]:
    seconds = median_us * 1e-6
    shape = case["shape"]
    dtype_bytes = 2 if case["dtype"] == "float16" else 4
    operator = case["operator"]
    if operator in {"vector_add", "relu_add"}:
        effective_bytes = 3 * shape[0] * dtype_bytes
        return {
            "effective_bytes": effective_bytes,
            "effective_gb_per_s": effective_bytes / seconds / 1e9,
        }
    if operator == "softmax":
        effective_bytes = 2 * shape[0] * shape[1] * dtype_bytes
        return {
            "effective_bytes": effective_bytes,
            "effective_gb_per_s": effective_bytes / seconds / 1e9,
        }
    if operator == "layer_norm":
        rows, hidden = shape
        effective_bytes = (2 * rows * hidden + 2 * hidden) * dtype_bytes
        return {
            "effective_bytes": effective_bytes,
            "effective_gb_per_s": effective_bytes / seconds / 1e9,
        }
    if operator == "matmul":
        rows, cols, reduction = shape
        flops = 2 * rows * cols * reduction
        return {"flops": flops, "tflops": flops / seconds / 1e12}
    if operator == "attention":
        heads, sequence, head_dim = shape
        flops = 4 * heads * sequence * sequence * head_dim
        return {"flops": flops, "tflops": flops / seconds / 1e12}
    return {}


def run_child(case_id: str, implementation: str, *, samples: int, warmup: int) -> None:
    import torch  # type: ignore[import-not-found]  # noqa: PLC0415

    case = CASES[case_id]
    reference, triton_candidate, accuracy_reference = build_callables(case, torch)
    function = reference if implementation == "pytorch" else triton_candidate

    neutral = torch.ones(1, device="cuda") + 1
    torch.cuda.synchronize()
    del neutral

    if implementation == "triton":
        expected = accuracy_reference()
        torch.cuda.synchronize()
    else:
        expected = None

    cold_start = time.perf_counter()
    actual = function()
    torch.cuda.synchronize()
    cold_start_ms = (time.perf_counter() - cold_start) * 1000.0
    if expected is None:
        expected = accuracy_reference()
        torch.cuda.synchronize()

    atol, rtol = tolerances(case)
    correctness = correctness_report(
        case["operator"], actual, expected, atol=atol, rtol=rtol, torch=torch
    )
    if correctness["status"] != "passed":
        payload = {
            "status": "correctness_error",
            "case_id": case_id,
            "implementation": implementation,
            "case": case,
            "cold_start_ms": cold_start_ms,
            "correctness": correctness,
        }
        print(RESULT_MARKER + json.dumps(payload, sort_keys=True), flush=True)
        raise SystemExit(2)

    del actual, expected
    gc.collect()
    torch.cuda.empty_cache()
    steady = steady_benchmark(
        function,
        inner=case["inner"],
        samples=samples,
        warmup=warmup,
        torch=torch,
    )
    median_us = steady["latency_us"]["median"]
    payload = {
        "status": "success",
        "case_id": case_id,
        "implementation": implementation,
        "case": case,
        "cold_start_ms": cold_start_ms,
        "correctness": correctness,
        "steady_state": steady,
        "metrics": rate_metrics(case, median_us),
    }
    print(RESULT_MARKER + json.dumps(payload, sort_keys=True), flush=True)


def parse_child_result(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    result: dict[str, Any] | None = None
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            result = json.loads(line.removeprefix(RESULT_MARKER))
    if result is None:
        signal_number = -completed.returncode if completed.returncode < 0 else None
        return {
            "status": "runtime_error",
            "returncode": completed.returncode,
            "shell_exit_code": shell_exit_code(completed.returncode),
            "signal": signal_number,
            "stdout": tail(completed.stdout),
            "stderr": tail(completed.stderr),
        }
    result.update(
        {
            "returncode": completed.returncode,
            "shell_exit_code": shell_exit_code(completed.returncode),
            "signal": -completed.returncode if completed.returncode < 0 else None,
            "stderr": tail(completed.stderr),
        }
    )
    return result


def environment_manifest() -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]  # noqa: PLC0415
    import triton  # type: ignore[import-not-found]  # noqa: PLC0415

    try:
        triton_package = version("triton")
    except PackageNotFoundError:
        triton_package = None
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
        "triton": triton.__version__,
        "triton_package": triton_package,
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": properties.total_memory,
        "driver": driver,
    }


def git_output(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def run_parent(args: argparse.Namespace) -> None:
    repository_path = str(REPOSITORY_ROOT.resolve())
    performance_location_ok = repository_path.startswith("/home/")
    if not performance_location_ok and not args.allow_mounted_path:
        raise SystemExit(
            "Formal performance runs must use a WSL /home/... checkout; "
            "pass --allow-mounted-path only for a non-reportable smoke run."
        )

    suite_case_ids = SMOKE_CASES if args.suite == "smoke" else FULL_CASES
    try:
        case_ids = select_case_ids(
            CASES,
            suite_case_ids,
            operators=tuple(args.operator),
            requested_case_ids=tuple(args.case_ids),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="trainscale-02-benchmark-") as cache_root:
        for case_id in case_ids:
            for implementation in ("pytorch", "triton"):
                child_environment = os.environ.copy()
                cache_name = f"{case_id}-{implementation}"
                child_environment.update(
                    {
                        "PYTHONFAULTHANDLER": "1",
                        "TRITON_CACHE_DIR": str(Path(cache_root) / cache_name / "triton"),
                        "TORCHINDUCTOR_CACHE_DIR": str(
                            Path(cache_root) / cache_name / "inductor"
                        ),
                    }
                )
                try:
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--child-case",
                            case_id,
                            "--child-implementation",
                            implementation,
                            "--samples",
                            str(args.samples),
                            "--warmup",
                            str(args.warmup),
                        ],
                        capture_output=True,
                        check=False,
                        cwd=REPOSITORY_ROOT,
                        env=child_environment,
                        text=True,
                        timeout=args.timeout_seconds,
                    )
                    result = parse_child_result(completed)
                except subprocess.TimeoutExpired as error:
                    result = {
                        "status": "timeout",
                        "returncode": None,
                        "shell_exit_code": None,
                        "signal": None,
                        "stdout": tail(error.stdout or ""),
                        "stderr": tail(error.stderr or ""),
                    }
                result.setdefault("case_id", case_id)
                result.setdefault("implementation", implementation)
                results.append(result)
                print(f"{case_id}/{implementation}: {result['status']}", flush=True)

    comparisons: list[dict[str, Any]] = []
    for case_id in case_ids:
        by_implementation = {
            result["implementation"]: result
            for result in results
            if result["case_id"] == case_id
        }
        reference = by_implementation["pytorch"]
        candidate = by_implementation["triton"]
        comparison: dict[str, Any] = {"case_id": case_id, "status": "unavailable"}
        if reference["status"] == "success" and candidate["status"] == "success":
            reference_us = reference["steady_state"]["latency_us"]["median"]
            candidate_us = candidate["steady_state"]["latency_us"]["median"]
            comparison.update(
                {
                    "status": "success",
                    "pytorch_median_us": reference_us,
                    "triton_median_us": candidate_us,
                    "triton_speedup_over_pytorch": reference_us / candidate_us,
                }
            )
        comparisons.append(comparison)

    failures = [result for result in results if result["status"] != "success"]
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(),
        "scope": "forward",
        "suite": args.suite,
        "selection": {
            "operators": args.operator,
            "requested_case_ids": args.case_ids,
            "selected_case_ids": list(case_ids),
        },
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "repository_root": repository_path,
        "performance_location_ok": performance_location_ok,
        "environment": environment_manifest(),
        "protocol": {
            "process_isolation": "one subprocess per case and implementation",
            "cold_timer": "perf_counter plus cuda synchronize with fresh Triton cache",
            "steady_timer": "torch.cuda.Event",
            "samples": args.samples,
            "warmup_calls": args.warmup,
            "allocation_note": "Logical operator output allocation is included.",
            "speedup_note": "Only steady-state medians are used for speedup.",
            "matmul_correctness_note": (
                "Both FP16 implementations are checked against FP32 matmul. "
                "For K >= 256, atol=0.025 accounts for long-reduction rounding; "
                "rtol remains 0.02."
            ),
        },
        "results": results,
        "comparisons": comparisons,
        "all_cases_passed": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(f"all_cases_passed={payload['all_cases_passed']}")
    raise SystemExit(0 if not failures else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--operator",
        action="append",
        choices=tuple(sorted({case["operator"] for case in CASES.values()})),
        default=[],
        help="Run every configured case for this operator; may be repeated.",
    )
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        choices=tuple(CASES),
        default=[],
        help="Run exactly this case; may be repeated and cannot be combined with --operator.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("02_gpu_kernels/results/triton_comparison.json"),
    )
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--allow-mounted-path", action="store_true")
    parser.add_argument("--child-case", choices=tuple(CASES), help=argparse.SUPPRESS)
    parser.add_argument(
        "--child-implementation",
        choices=("pytorch", "triton"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.warmup < 0 or args.timeout_seconds <= 0:
        raise SystemExit("samples/timeout must be positive and warmup must be non-negative")
    child_requested = args.child_case is not None or args.child_implementation is not None
    if child_requested:
        if args.child_case is None or args.child_implementation is None:
            raise SystemExit("Both hidden child arguments are required")
        run_child(
            args.child_case,
            args.child_implementation,
            samples=args.samples,
            warmup=args.warmup,
        )
        return
    run_parent(args)


if __name__ == "__main__":
    main()
