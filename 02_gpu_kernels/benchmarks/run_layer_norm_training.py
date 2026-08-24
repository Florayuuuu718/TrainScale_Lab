"""Crash-isolated LayerNorm forward/backward correctness and performance sweep."""

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
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))

from benchmark_contract import load_cases, percentile, validate_result_record  # noqa: E402

CASES = load_cases(MODULE_ROOT / "configs" / "layer_norm_training.toml")
RESULT_MARKER = "TRAINSCALE_LAYER_NORM_RESULT="
SEED = 20260824


def dtype_from_name(torch: Any, name: str) -> Any:
    return {"float16": torch.float16, "float32": torch.float32}[name]


def build_callables(
    case: dict[str, Any], phase: str, implementation: str, torch: Any
) -> tuple[Any, Any]:
    from trainscale_kernels import layer_norm, layer_norm_backward  # noqa: PLC0415

    torch.manual_seed(SEED)
    rows, hidden = case["shape"]
    dtype = dtype_from_name(torch, case["dtype"])
    eps = case["eps"]
    x = torch.randn((rows, hidden), device="cuda", dtype=dtype)
    weight = torch.randn(hidden, device="cuda", dtype=dtype)
    bias = torch.randn(hidden, device="cuda", dtype=dtype)
    grad = torch.randn_like(x)

    def pytorch_forward() -> Any:
        return torch.native_layer_norm(x, (hidden,), weight, bias, eps)

    def triton_forward() -> Any:
        return layer_norm(x, weight, bias, eps)
    if phase == "forward":
        expected = pytorch_forward()
        function = pytorch_forward if implementation == "pytorch" else triton_forward
        return function, expected

    pytorch_saved = pytorch_forward()
    triton_saved = triton_forward()
    torch.cuda.synchronize()

    def pytorch_backward() -> Any:
        return torch.ops.aten.native_layer_norm_backward.default(
            grad,
            x,
            [hidden],
            pytorch_saved[1],
            pytorch_saved[2],
            weight,
            bias,
            [True, True, True],
        )

    def triton_backward() -> Any:
        return layer_norm_backward(grad, x, weight, triton_saved[1], triton_saved[2])

    expected = pytorch_backward()
    function = pytorch_backward if implementation == "pytorch" else triton_backward
    return function, expected


def correctness_report(actual: Any, expected: Any, case: dict[str, Any], torch: Any) -> Any:
    tensors = actual if isinstance(actual, tuple) else (actual,)
    references = expected if isinstance(expected, tuple) else (expected,)
    if len(tensors) == 3:
        tensors = (tensors[0], tensors[1].reshape(-1), tensors[2].reshape(-1))
        references = (
            references[0],
            references[1].reshape(-1),
            references[2].reshape(-1),
        )
    atol = 1e-4 if case["dtype"] == "float32" else 4e-2
    rtol = atol
    maximum = 0.0
    try:
        for value, reference in zip(tensors, references, strict=True):
            difference = (value.float() - reference.float()).abs()
            maximum = max(maximum, difference.max().item())
            torch.testing.assert_close(
                value, reference, atol=atol, rtol=rtol, check_dtype=False
            )
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


def benchmark(function: Any, case: dict[str, Any], args: argparse.Namespace, torch: Any) -> Any:
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


def run_child(args: argparse.Namespace) -> None:
    import torch  # type: ignore[import-not-found]  # noqa: PLC0415

    case = CASES[args.child_case]
    function, expected = build_callables(
        case, args.child_phase, args.child_implementation, torch
    )
    torch.cuda.synchronize()
    cold_start = time.perf_counter()
    actual = function()
    torch.cuda.synchronize()
    cold_start_ms = (time.perf_counter() - cold_start) * 1000.0
    correctness = correctness_report(actual, expected, case, torch)
    if correctness["status"] != "passed":
        print(
            RESULT_MARKER
            + json.dumps(
                {
                    "case_id": args.child_case,
                    "phase": args.child_phase,
                    "implementation": args.child_implementation,
                    "status": "correctness_error",
                    "correctness": correctness,
                }
            ),
            flush=True,
        )
        raise SystemExit(2)
    del actual, expected
    gc.collect()
    steady = benchmark(function, case, args, torch)
    rows, hidden = case["shape"]
    dtype_bytes = 2 if case["dtype"] == "float16" else 4
    payload = {
        "case_id": args.child_case,
        "phase": args.child_phase,
        "implementation": args.child_implementation,
        "status": "success",
        "case": case,
        "cold_start_ms": cold_start_ms,
        "correctness": correctness,
        "steady_state": steady,
        "metrics": {
            "tensor_bytes": rows * hidden * dtype_bytes,
            "saved_mean_rstd_bytes": rows * 2 * 4,
            "parameter_bytes": hidden * 2 * dtype_bytes,
        },
    }
    print(RESULT_MARKER + json.dumps(payload), flush=True)


def parse_result(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
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


def run_parent(args: argparse.Namespace) -> None:
    import torch  # type: ignore[import-not-found]  # noqa: PLC0415
    import triton  # type: ignore[import-not-found]  # noqa: PLC0415

    repository_path = str(REPOSITORY_ROOT.resolve())
    if not repository_path.startswith("/home/") and not args.allow_mounted_path:
        raise SystemExit("Formal performance runs require a /home/... checkout")
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="trainscale-layer-norm-") as cache_root:
        for case_id in CASES:
            for phase in ("forward", "backward"):
                for implementation in ("pytorch", "triton"):
                    environment = os.environ.copy()
                    environment.update(
                        {
                            "TRITON_CACHE_DIR": str(
                                Path(cache_root) / case_id / phase / implementation
                            ),
                            "PYTHONFAULTHANDLER": "1",
                        }
                    )
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--child-case",
                            case_id,
                            "--child-phase",
                            phase,
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
                        env=environment,
                        text=True,
                        timeout=args.timeout_seconds,
                    )
                    result = parse_result(completed)
                    result.setdefault("case_id", case_id)
                    result.setdefault("phase", phase)
                    result.setdefault("implementation", implementation)
                    results.append(result)
                    print(
                        f"{case_id}/{phase}/{implementation}: {result['status']}",
                        flush=True,
                    )

    comparisons: list[dict[str, Any]] = []
    for case_id in CASES:
        for phase in ("forward", "backward"):
            selected = [
                result
                for result in results
                if result["case_id"] == case_id and result["phase"] == phase
            ]
            by_name = {result["implementation"]: result for result in selected}
            comparison: dict[str, Any] = {
                "case_id": case_id,
                "phase": phase,
                "status": "unavailable",
            }
            if all(result["status"] == "success" for result in selected):
                pytorch_us = by_name["pytorch"]["steady_state"]["latency_us"]["median"]
                triton_us = by_name["triton"]["steady_state"]["latency_us"]["median"]
                comparison.update(
                    {
                        "status": "success",
                        "pytorch_median_us": pytorch_us,
                        "triton_median_us": triton_us,
                        "triton_speedup_over_pytorch": pytorch_us / triton_us,
                    }
                )
            comparisons.append(comparison)

    for result in results:
        if result["status"] == "success":
            validate_result_record(result)
    failures = [result for result in results if result["status"] != "success"]
    properties = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(),
        "scope": "LayerNorm forward and backward",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "repository_root": repository_path,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "triton": triton.__version__,
            "gpu": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "driver": driver,
        },
        "protocol": {
            "process_isolation": "one subprocess per case/phase/implementation",
            "allocation_note": "Logical output allocation is included for both implementations.",
            "backward_note": (
                "Backward uses saved mean/rstd from the matching forward implementation."
            ),
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("02_gpu_kernels/results/layer_norm_training.json"),
    )
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--allow-mounted-path", action="store_true")
    parser.add_argument("--child-case", choices=tuple(CASES), help=argparse.SUPPRESS)
    parser.add_argument(
        "--child-phase", choices=("forward", "backward"), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--child-implementation", choices=("pytorch", "triton"), help=argparse.SUPPRESS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.warmup < 0 or args.timeout_seconds <= 0:
        raise SystemExit("samples/timeout must be positive and warmup non-negative")
    child = any(
        value is not None
        for value in (args.child_case, args.child_phase, args.child_implementation)
    )
    if child:
        if None in (args.child_case, args.child_phase, args.child_implementation):
            raise SystemExit("all child arguments are required")
        run_child(args)
        return
    run_parent(args)


if __name__ == "__main__":
    main()
