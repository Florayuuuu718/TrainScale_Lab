"""Kernel-only PyTorch/Triton/CUDA C++ comparison for Vector Add and Softmax."""

from __future__ import annotations

import argparse
import gc
import hashlib
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

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))

from benchmark_contract import load_cases, percentile, validate_result_record  # noqa: E402

CASES = load_cases(MODULE_ROOT / "configs" / "cuda_comparison.toml")
RESULT_MARKER = "TRAINSCALE_CUDA_COMPARISON_RESULT="


def dtype_from_name(torch: Any, name: str) -> Any:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_python_callables(case: dict[str, Any], torch: Any) -> tuple[Any, Any, Any, Any]:
    from trainscale_kernels import softmax, softmax_baseline, vector_add  # noqa: PLC0415

    dtype = dtype_from_name(torch, case["dtype"])
    if case["operator"] == "vector_add":
        size = case["shape"][0]
        indices = torch.arange(size, device="cuda", dtype=torch.float32)
        x = torch.sin(indices * 0.001).to(dtype)
        y = torch.cos(indices * 0.0013).to(dtype)
        output = torch.empty_like(x)

        def reference() -> Any:
            return torch.add(x, y, out=output)

        def candidate() -> Any:
            return vector_add(x, y, out=output)

        return reference, candidate, candidate, lambda: x + y

    rows, cols = case["shape"]
    indices = torch.arange(rows * cols, device="cuda", dtype=torch.float32).reshape(rows, cols)
    row_indices = torch.arange(rows, device="cuda", dtype=torch.float32).reshape(rows, 1)
    x = torch.sin(indices * 0.013) * 20.0 + torch.cos(row_indices * 0.17)
    output = torch.empty_like(x)

    def reference() -> Any:
        return torch.ops.aten._softmax.out(x, -1, False, out=output)

    def candidate() -> Any:
        return softmax(x, out=output)

    def baseline_candidate() -> Any:
        return softmax_baseline(x, out=output)

    return reference, candidate, baseline_candidate, lambda: torch.softmax(x, dim=-1)


def correctness_report(
    actual: Any, expected: Any, case: dict[str, Any], torch: Any
) -> dict[str, Any]:
    if case["operator"] == "softmax":
        atol, rtol = 2e-6, 2e-5
    elif case["dtype"] == "float32":
        atol, rtol = 1e-6, 1e-6
    else:
        atol, rtol = 1e-3, 1e-3
    difference = (actual.float() - expected.float()).abs()
    maximum = difference.max().item()
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as error:
        return {
            "status": "failed",
            "atol": atol,
            "rtol": rtol,
            "max_absolute_error": maximum,
            "error": str(error)[-2000:],
        }
    return {
        "status": "passed",
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error": maximum,
    }


def steady_benchmark(
    function: Any, case: dict[str, Any], args: argparse.Namespace, torch: Any
) -> Any:
    for _ in range(args.warmup):
        function()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(case["inner"]):
            function()
        stop.record()
        stop.synchronize()
        values.append(start.elapsed_time(stop) * 1000.0 / case["inner"])
    return {
        "timer": "torch.cuda.Event",
        "inner_iterations": case["inner"],
        "samples": args.samples,
        "warmup_calls": args.warmup,
        "latency_us": {
            "median": statistics.median(values),
            "p10": percentile(values, 0.1),
            "p90": percentile(values, 0.9),
        },
    }


def metrics(case: dict[str, Any], median_us: float) -> dict[str, Any]:
    elements = 1
    for value in case["shape"]:
        elements *= value
    dtype_bytes = 2 if case["dtype"] != "float32" else 4
    factor = 3 if case["operator"] == "vector_add" else 2
    effective_bytes = factor * elements * dtype_bytes
    return {
        "effective_bytes": effective_bytes,
        "effective_gb_per_s": effective_bytes / (median_us * 1e3),
    }


def run_python_child(args: argparse.Namespace) -> None:
    import torch  # type: ignore[import-not-found]  # noqa: PLC0415

    case = CASES[args.child_case]
    reference, candidate, baseline_candidate, accuracy_reference = build_python_callables(
        case, torch
    )
    if args.child_implementation == "pytorch":
        function = reference
    elif args.child_implementation == "triton_baseline":
        function = baseline_candidate
    else:
        function = candidate
    torch.cuda.synchronize()
    expected = accuracy_reference()
    torch.cuda.synchronize()
    cold_start = time.perf_counter()
    actual = function()
    torch.cuda.synchronize()
    cold_start_ms = (time.perf_counter() - cold_start) * 1000.0
    correctness = correctness_report(actual, expected, case, torch)
    if correctness["status"] != "passed":
        payload = {
            "case_id": args.child_case,
            "implementation": args.child_implementation,
            "status": "correctness_error",
            "correctness": correctness,
        }
        print(RESULT_MARKER + json.dumps(payload), flush=True)
        raise SystemExit(2)
    del actual, expected
    gc.collect()
    steady = steady_benchmark(function, case, args, torch)
    payload = {
        "case_id": args.child_case,
        "implementation": args.child_implementation,
        "status": "success",
        "case": case,
        "cold_start_ms": cold_start_ms,
        "correctness": correctness,
        "steady_state": steady,
        "metrics": metrics(case, steady["latency_us"]["median"]),
    }
    print(RESULT_MARKER + json.dumps(payload), flush=True)


def parse_python_result(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            result = json.loads(line.removeprefix(RESULT_MARKER))
            result["returncode"] = completed.returncode
            result["stderr"] = completed.stderr[-4000:]
            return result
    return {
        "status": "runtime_error",
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def cuda_command(
    binary: Path, case: dict[str, Any], variant: str, args: argparse.Namespace
) -> list[str]:
    command = [
        str(binary),
        "--operator",
        case["operator"],
        "--variant",
        variant,
        "--dtype",
        case["dtype"],
        "--samples",
        str(args.samples),
        "--warmup",
        str(args.warmup),
        "--inner",
        str(case["inner"]),
    ]
    if case["operator"] == "vector_add":
        command.extend(["--size", str(case["shape"][0])])
    else:
        command.extend(["--rows", str(case["shape"][0]), "--cols", str(case["shape"][1])])
    return command


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
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "triton_package": triton_package,
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "driver": driver,
    }


def run_parent(args: argparse.Namespace) -> None:
    repository_path = str(REPOSITORY_ROOT.resolve())
    if not repository_path.startswith("/home/") and not args.allow_mounted_path:
        raise SystemExit("Formal performance runs require a /home/... checkout")
    binary = args.cuda_binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"CUDA benchmark binary not found: {binary}")
    binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="trainscale-02-cuda-compare-") as cache_root:
        for case_id, case in CASES.items():
            python_implementations = ["pytorch", "triton"]
            if case["operator"] == "softmax":
                python_implementations.insert(1, "triton_baseline")
            for implementation in python_implementations:
                environment = os.environ.copy()
                environment.update(
                    {
                        "TRITON_CACHE_DIR": str(Path(cache_root) / case_id / implementation),
                        "PYTHONFAULTHANDLER": "1",
                    }
                )
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
                        "--cuda-binary",
                        str(binary),
                    ],
                    capture_output=True,
                    check=False,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    text=True,
                    timeout=args.timeout_seconds,
                )
                result = parse_python_result(completed)
                result.setdefault("case_id", case_id)
                result.setdefault("implementation", implementation)
                results.append(result)
                print(f"{case_id}/{implementation}: {result['status']}", flush=True)

            for variant in ("baseline", "optimized"):
                completed = subprocess.run(
                    cuda_command(binary, case, variant, args),
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=args.timeout_seconds,
                )
                try:
                    result = json.loads(completed.stdout.splitlines()[-1])
                except (IndexError, json.JSONDecodeError):
                    result = {
                        "status": "runtime_error",
                        "stdout": completed.stdout[-4000:],
                        "stderr": completed.stderr[-4000:],
                    }
                result.update(
                    {
                        "case_id": case_id,
                        "implementation": f"cuda_{variant}",
                        "case": case,
                        "returncode": completed.returncode,
                    }
                )
                if result["status"] == "success":
                    result["steady_state"].update(
                        {
                            "timer": "cudaEvent",
                            "inner_iterations": case["inner"],
                            "samples": args.samples,
                            "warmup_calls": args.warmup,
                        }
                    )
                results.append(result)
                print(f"{case_id}/cuda_{variant}: {result['status']}", flush=True)

    comparisons: list[dict[str, Any]] = []
    for case_id in CASES:
        case_results = {
            result["implementation"]: result
            for result in results
            if result["case_id"] == case_id
        }
        comparison: dict[str, Any] = {"case_id": case_id, "status": "unavailable"}
        if all(result["status"] == "success" for result in case_results.values()):
            medians = {
                name: result["steady_state"]["latency_us"]["median"]
                for name, result in case_results.items()
            }
            baseline = medians["pytorch"]
            comparison = {
                "case_id": case_id,
                "status": "success",
                "median_us": medians,
                "speedup_over_pytorch": {
                    name: baseline / value for name, value in medians.items() if name != "pytorch"
                },
            }
        comparisons.append(comparison)

    for result in results:
        if result["status"] == "success":
            validate_result_record(result)
    failures = [result for result in results if result["status"] != "success"]
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(),
        "scope": "kernel-only Vector Add and Softmax",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "repository_root": repository_path,
        "environment": environment_manifest(),
        "cuda_cpp_build": {
            "binary": str(binary),
            "binary_sha256": binary_sha256,
            "toolkit": args.cuda_toolkit,
            "nvcc_flags": args.cuda_build_flag,
            "source": "02_gpu_kernels/cuda/kernel_bench.cu",
        },
        "protocol": {
            "input_formula": (
                "Vector: sin(i*0.001), cos(i*0.0013). Softmax: "
                "sin(i*0.013)*20 + cos(row*0.17)."
            ),
            "same_domain_note": (
                "All implementations use the same deterministic formula and shape/dtype."
            ),
            "allocation_note": (
                "Device allocations and host-device copies are excluded for all four paths."
            ),
            "correctness_before_performance": True,
            "samples": args.samples,
            "warmup_calls": args.warmup,
        },
        "cases": CASES,
        "results": results,
        "comparisons": comparisons,
        "all_cases_passed": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(f"all_cases_passed={not failures}")
    raise SystemExit(0 if not failures else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-binary", type=Path, required=True)
    parser.add_argument("--cuda-toolkit", default="unknown")
    parser.add_argument("--cuda-build-flag", action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("02_gpu_kernels/results/cuda_triton_comparison.json"),
    )
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--allow-mounted-path", action="store_true")
    parser.add_argument("--child-case", choices=tuple(CASES), help=argparse.SUPPRESS)
    parser.add_argument(
        "--child-implementation",
        choices=("pytorch", "triton_baseline", "triton"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.warmup < 0 or args.timeout_seconds <= 0:
        raise SystemExit("samples/timeout must be positive and warmup non-negative")
    if args.child_case is not None or args.child_implementation is not None:
        if args.child_case is None or args.child_implementation is None:
            raise SystemExit("both child arguments are required")
        run_python_child(args)
        return
    run_parent(args)


if __name__ == "__main__":
    main()
