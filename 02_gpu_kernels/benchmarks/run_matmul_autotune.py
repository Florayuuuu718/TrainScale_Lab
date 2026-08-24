"""Measure and record a finite Triton MatMul tile/warp search space."""

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

from benchmark_contract import (  # noqa: E402
    load_cases,
    load_matmul_candidates,
    percentile,
    validate_result_record,
)

CASES = load_cases(MODULE_ROOT / "configs" / "matmul_autotune_cases.toml")
CANDIDATES = load_matmul_candidates(MODULE_ROOT / "configs" / "matmul_candidates.toml")
IMPLEMENTATIONS = ("pytorch", *CANDIDATES)
RESULT_MARKER = "TRAINSCALE_MATMUL_TUNE_RESULT="
SEED = 20260824


def build_callable(case: dict[str, Any], implementation: str, torch: Any) -> tuple[Any, Any]:
    from trainscale_kernels import matmul_configured  # noqa: PLC0415

    torch.manual_seed(SEED)
    rows, cols, reduction = case["shape"]
    left = torch.randn((rows, reduction), device="cuda", dtype=torch.float16)
    right = torch.randn((reduction, cols), device="cuda", dtype=torch.float16)
    expected = left.float() @ right.float()
    if implementation == "pytorch":
        return lambda: left @ right, expected
    candidate = CANDIDATES[implementation]
    return (
        lambda: matmul_configured(left, right, **candidate),
        expected,
    )


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
    function, expected = build_callable(case, args.child_implementation, torch)
    torch.cuda.synchronize()
    cold_start = time.perf_counter()
    actual = function()
    torch.cuda.synchronize()
    cold_start_ms = (time.perf_counter() - cold_start) * 1000.0
    reduction = case["shape"][2]
    atol = 2.5e-2 if reduction >= 256 else 2e-2
    difference = (actual.float() - expected).abs()
    maximum = difference.max().item()
    try:
        torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=2e-2)
    except AssertionError as error:
        print(
            RESULT_MARKER
            + json.dumps(
                {
                    "case_id": args.child_case,
                    "implementation": args.child_implementation,
                    "status": "correctness_error",
                    "correctness": {
                        "status": "failed",
                        "atol": atol,
                        "rtol": 2e-2,
                        "max_absolute_error": maximum,
                        "error": str(error)[-2000:],
                    },
                }
            ),
            flush=True,
        )
        raise SystemExit(2) from None
    del actual, expected
    gc.collect()
    steady = benchmark(function, case, args, torch)
    rows, cols, reduction = case["shape"]
    flops = 2 * rows * cols * reduction
    median_us = steady["latency_us"]["median"]
    payload = {
        "case_id": args.child_case,
        "implementation": args.child_implementation,
        "candidate": CANDIDATES.get(args.child_implementation),
        "status": "success",
        "case": case,
        "cold_start_ms": cold_start_ms,
        "correctness": {
            "status": "passed",
            "atol": atol,
            "rtol": 2e-2,
            "max_absolute_error": maximum,
            "reference": "fp32_matmul",
        },
        "steady_state": steady,
        "metrics": {"flops": flops, "tflops": flops / (median_us * 1e6)},
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
    with tempfile.TemporaryDirectory(prefix="trainscale-matmul-tune-") as cache_root:
        for case_id in CASES:
            for implementation in IMPLEMENTATIONS:
                environment = os.environ.copy()
                environment.update(
                    {
                        "TRITON_CACHE_DIR": str(
                            Path(cache_root) / case_id / implementation
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
                result.setdefault("implementation", implementation)
                results.append(result)
                print(f"{case_id}/{implementation}: {result['status']}", flush=True)

    selections: list[dict[str, Any]] = []
    for case_id in CASES:
        selected = [result for result in results if result["case_id"] == case_id]
        successful_candidates = [
            result
            for result in selected
            if result["implementation"] != "pytorch" and result["status"] == "success"
        ]
        if not successful_candidates:
            selections.append({"case_id": case_id, "status": "unavailable"})
            continue
        best = min(
            successful_candidates,
            key=lambda result: result["steady_state"]["latency_us"]["median"],
        )
        pytorch_result = next(
            result for result in selected if result["implementation"] == "pytorch"
        )
        best_us = best["steady_state"]["latency_us"]["median"]
        pytorch_us = pytorch_result["steady_state"]["latency_us"]["median"]
        selections.append(
            {
                "case_id": case_id,
                "status": "success",
                "selected_candidate": best["implementation"],
                "selected_config": best["candidate"],
                "selected_median_us": best_us,
                "pytorch_median_us": pytorch_us,
                "selected_speedup_over_pytorch": pytorch_us / best_us,
                "candidate_ranking": [
                    {
                        "candidate": result["implementation"],
                        "median_us": result["steady_state"]["latency_us"]["median"],
                    }
                    for result in sorted(
                        successful_candidates,
                        key=lambda result: result["steady_state"]["latency_us"]["median"],
                    )
                ],
            }
        )

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
        "scope": "finite MatMul tile/warp search",
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
            "search_type": "exhaustive finite teaching search",
            "process_isolation": "one subprocess per case and candidate",
            "selection_rule": "lowest steady-state median among correctness-passing candidates",
            "correctness_reference": "FP32 matmul for PyTorch and every Triton candidate",
            "samples": args.samples,
            "warmup_calls": args.warmup,
        },
        "cases": CASES,
        "candidates": CANDIDATES,
        "results": results,
        "selections": selections,
        "all_candidates_passed": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(f"all_candidates_passed={not failures}")
    raise SystemExit(0 if not failures else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("02_gpu_kernels/results/matmul_autotune.json"),
    )
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--allow-mounted-path", action="store_true")
    parser.add_argument("--child-case", choices=tuple(CASES), help=argparse.SUPPRESS)
    parser.add_argument(
        "--child-implementation", choices=IMPLEMENTATIONS, help=argparse.SUPPRESS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.warmup < 0 or args.timeout_seconds <= 0:
        raise SystemExit("samples/timeout must be positive and warmup non-negative")
    if args.child_case is not None or args.child_implementation is not None:
        if args.child_case is None or args.child_implementation is None:
            raise SystemExit("both child arguments are required")
        run_child(args)
        return
    run_parent(args)


if __name__ == "__main__":
    main()
