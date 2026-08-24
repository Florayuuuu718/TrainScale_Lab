"""Probe the module 02 GPU stack without letting a native crash kill the runner.

Each executable CUDA/Triton check runs in its own Python subprocess.  This matters
because an invalid Triton launch can terminate the process with SIGSEGV instead of
raising a catchable Python exception.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULT_MARKER = "TRAIN_SCALE_PROBE_RESULT="


ENVIRONMENT_PROBE = f"""
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import torch

try:
    import triton
    triton_version = triton.__version__
except Exception as error:
    triton_version = f"error: {{type(error).__name__}}: {{error}}"
try:
    triton_package_version = version("triton")
except PackageNotFoundError:
    triton_package_version = None

cuda_available = torch.cuda.is_available()
payload = {{
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "triton": triton_version,
    "triton_package": triton_package_version,
    "cuda_available": cuda_available,
}}
if cuda_available:
    properties = torch.cuda.get_device_properties(0)
    payload.update(
        {{
            "gpu": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
        }}
    )
print({RESULT_MARKER!r} + json.dumps(payload, sort_keys=True))
"""


CUDA_EAGER_PROBE = """
import torch

x = torch.arange(4097, device="cuda", dtype=torch.float32)
y = torch.ones_like(x)
actual = x + y
torch.cuda.synchronize()
torch.testing.assert_close(actual, x.cpu().cuda() + 1)
print("CUDA eager forward passed")
"""


TORCH_COMPILE_PROBE = """
import torch

x = torch.linspace(-2, 2, 4097, device="cuda")
y = torch.full_like(x, 0.25)
compiled = torch.compile(lambda left, right: torch.relu(left + right), fullgraph=True)
actual = compiled(x, y)
torch.cuda.synchronize()
torch.testing.assert_close(actual, torch.relu(x + y))
actual = compiled(x, y)
torch.cuda.synchronize()
torch.testing.assert_close(actual, torch.relu(x + y))
print("torch.compile cold and warm calls passed")
"""


TRITON_SINGLE_LOAD_PROBE = """
import os
import sys

import torch

repository_root = os.environ["TRAINSCALE_REPOSITORY_ROOT"]
sys.path.insert(0, os.path.join(repository_root, "02_gpu_kernels"))
from trainscale_kernels import softmax

x = torch.randn((32, 127), device="cuda", dtype=torch.float32) * 20
actual = softmax(x)
torch.cuda.synchronize()
torch.testing.assert_close(actual, torch.softmax(x, dim=-1), atol=2e-6, rtol=2e-5)
print("repository Triton single-input-load softmax passed")
"""


TRITON_VECTOR_ADD_PROBE = """
import os
import sys

import torch

repository_root = os.environ["TRAINSCALE_REPOSITORY_ROOT"]
sys.path.insert(0, os.path.join(repository_root, "02_gpu_kernels"))
from trainscale_kernels import vector_add

x = torch.arange(4097, device="cuda", dtype=torch.float32)
y = torch.linspace(-1, 1, 4097, device="cuda")
actual = vector_add(x, y, block_size=128)
torch.cuda.synchronize()
torch.testing.assert_close(actual, x + y)
print("repository Triton two-load vector add passed")
"""


PROBES = {
    "environment": ENVIRONMENT_PROBE,
    "cuda_eager": CUDA_EAGER_PROBE,
    "torch_compile": TORCH_COMPILE_PROBE,
    "triton_single_load": TRITON_SINGLE_LOAD_PROBE,
    "triton_vector_add": TRITON_VECTOR_ADD_PROBE,
}


def _tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[-limit:]


def _shell_exit_code(returncode: int) -> int:
    """Translate subprocess' negative POSIX signal code to the shell convention."""

    return 128 - returncode if returncode < 0 else returncode


def run_python_probe(
    name: str,
    source: str,
    *,
    environment: dict[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "name": name,
            "status": "timeout",
            "returncode": None,
            "shell_exit_code": None,
            "signal": None,
            "stdout": _tail(error.stdout or ""),
            "stderr": _tail(error.stderr or ""),
        }

    signal_number = -completed.returncode if completed.returncode < 0 else None
    result: dict[str, Any] = {
        "name": name,
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "shell_exit_code": _shell_exit_code(completed.returncode),
        "signal": signal_number,
        "stdout": _tail(completed.stdout.strip()),
        "stderr": _tail(completed.stderr.strip()),
    }
    if name == "environment" and completed.returncode == 0:
        for line in completed.stdout.splitlines():
            if line.startswith(RESULT_MARKER):
                result["details"] = json.loads(line.removeprefix(RESULT_MARKER))
                break
    return result


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "output": f"{type(error).__name__}: {error}"}
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return {"returncode": completed.returncode, "output": _tail(output.strip())}


def nvidia_smi_report() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    wsl_executable = Path("/usr/lib/wsl/lib/nvidia-smi")
    if executable is None and wsl_executable.is_file():
        executable = str(wsl_executable)
    if executable is None:
        return {"returncode": None, "output": "nvidia-smi was not found"}
    return command_output(
        [
            executable,
            "--query-gpu=name,driver_version,compute_cap,memory.total",
            "--format=csv,noheader",
        ]
    )


def toolkit_report(
    *,
    require_nvcc: bool,
    target_gpu_code: str | None,
    nvcc_flags: list[str],
) -> dict[str, Any]:
    executable = shutil.which("nvcc")
    if executable is None:
        return {
            "status": "failed" if require_nvcc else "optional_not_installed",
            "nvcc": None,
            "version": None,
            "gpu_codes": None,
            "target_gpu_code": target_gpu_code,
            "smoke_compile": None,
            "smoke_run": None,
        }
    version = command_output([executable, "--version"])
    gpu_codes = command_output([executable, "--list-gpu-code"])
    basic_checks_passed = version["returncode"] == 0 and gpu_codes["returncode"] == 0
    target_supported = bool(
        target_gpu_code
        and gpu_codes["returncode"] == 0
        and target_gpu_code in gpu_codes["output"].splitlines()
    )
    smoke_compile: dict[str, Any] | None = None
    smoke_run: dict[str, Any] | None = None
    if require_nvcc and basic_checks_passed and target_supported:
        source = REPOSITORY_ROOT / "02_gpu_kernels" / "cuda" / "smoke_vector_add.cu"
        with tempfile.TemporaryDirectory(prefix="trainscale-02-nvcc-") as build_directory:
            executable_path = str(Path(build_directory) / "smoke_vector_add")
            smoke_compile = command_output(
                [
                    executable,
                    "-std=c++17",
                    "-O2",
                    f"-arch={target_gpu_code}",
                    *nvcc_flags,
                    str(source),
                    "-o",
                    executable_path,
                ]
            )
            if smoke_compile["returncode"] == 0:
                smoke_run = command_output([executable_path])

    smoke_passed = bool(
        smoke_compile
        and smoke_compile["returncode"] == 0
        and smoke_run
        and smoke_run["returncode"] == 0
    )
    passed = basic_checks_passed and (not require_nvcc or (target_supported and smoke_passed))
    return {
        "status": "passed" if passed else "failed",
        "nvcc": executable,
        "version": version,
        "gpu_codes": gpu_codes,
        "target_gpu_code": target_gpu_code,
        "target_supported": target_supported,
        "nvcc_flags": nvcc_flags,
        "smoke_compile": smoke_compile,
        "smoke_run": smoke_run,
    }


def recommended_route(probes: dict[str, dict[str, Any]]) -> str:
    if probes["cuda_eager"]["status"] != "passed":
        return "fix_driver_or_pytorch_cuda_before_module_02"
    triton_names = ("torch_compile", "triton_single_load", "triton_vector_add")
    if all(probes[name]["status"] == "passed" for name in triton_names):
        return "current_environment_is_ready_for_triton_correctness_tests"
    return "use_the_isolated_module_02_compatibility_environment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run crash-isolated CUDA/Triton capability checks for module 02."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output. Omit it for a read-only smoke run.",
    )
    parser.add_argument(
        "--require-nvcc",
        action="store_true",
        help="Compile and run the CUDA C++ smoke; make any Toolkit failure fatal.",
    )
    parser.add_argument(
        "--nvcc-flag",
        action="append",
        default=[],
        metavar="FLAG",
        help="Extra nvcc flag for a documented compatibility route; repeat as needed.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")

    with tempfile.TemporaryDirectory(prefix="trainscale-02-probe-") as cache_root:
        child_environment = os.environ.copy()
        child_environment.update(
            {
                "PYTHONFAULTHANDLER": "1",
                "TRAINSCALE_REPOSITORY_ROOT": str(REPOSITORY_ROOT),
                "TRITON_CACHE_DIR": str(Path(cache_root) / "triton"),
                "TORCHINDUCTOR_CACHE_DIR": str(Path(cache_root) / "inductor"),
            }
        )
        probe_results = {
            name: run_python_probe(
                name,
                source,
                environment=child_environment,
                timeout_seconds=args.timeout_seconds,
            )
            for name, source in PROBES.items()
        }

    environment_details = probe_results["environment"].get("details", {})
    capability = environment_details.get("compute_capability")
    target_gpu_code = (
        f"sm_{capability[0]}{capability[1]}"
        if isinstance(capability, list) and len(capability) == 2
        else None
    )
    toolkit = toolkit_report(
        require_nvcc=args.require_nvcc,
        target_gpu_code=target_gpu_code,
        nvcc_flags=args.nvcc_flag,
    )
    required_names = tuple(PROBES)
    probes_passed = all(probe_results[name]["status"] == "passed" for name in required_names)
    toolkit_passed = toolkit["status"] == "passed" or not args.require_nvcc
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(),
        "host_platform": platform.platform(),
        "working_directory": str(Path.cwd()),
        "repository_root": str(REPOSITORY_ROOT),
        "performance_location_ok": str(REPOSITORY_ROOT).startswith("/home/"),
        "nvidia_smi": nvidia_smi_report(),
        "repository": {
            "commit": command_output(["git", "rev-parse", "HEAD"]),
            "status": command_output(["git", "status", "--short"]),
        },
        "probes": probe_results,
        "cuda_toolkit": toolkit,
        "recommended_route": recommended_route(probe_results),
        "all_required_checks_passed": probes_passed and toolkit_passed,
    }

    for name in required_names:
        result = probe_results[name]
        code = result["shell_exit_code"]
        suffix = "" if code in (None, 0) else f" (shell exit {code})"
        print(f"{name}: {result['status']}{suffix}")
    print(f"cuda_toolkit: {toolkit['status']}")
    print(f"recommended_route={payload['recommended_route']}")
    print(f"all_required_checks_passed={payload['all_required_checks_passed']}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(args.output)

    raise SystemExit(0 if payload["all_required_checks_passed"] else 1)


if __name__ == "__main__":
    main()
