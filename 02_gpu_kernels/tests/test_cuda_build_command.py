from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def load_build_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "cuda" / "build_cuda_bench.py"
    spec = importlib.util.spec_from_file_location("build_cuda_bench", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_build_command_targets_real_architecture_and_keeps_flags() -> None:
    module = load_build_module()
    command = module.build_command(
        source=Path("kernel_bench.cu"),
        output=Path("kernel_bench"),
        architecture="sm_120",
        nvcc="/usr/local/cuda-13.0/bin/nvcc",
        extra_flags=["-U_GNU_SOURCE", "-D_DEFAULT_SOURCE"],
    )
    assert "-arch=sm_120" in command
    assert "-U_GNU_SOURCE" in command
    assert command[-2:] == ["-o", "kernel_bench"]
