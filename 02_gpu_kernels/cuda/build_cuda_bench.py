"""Compile the standalone module 02 CUDA C++ benchmark."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def build_command(
    *, source: Path, output: Path, architecture: str, nvcc: str, extra_flags: list[str]
) -> list[str]:
    return [
        nvcc,
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        f"-arch={architecture}",
        *extra_flags,
        str(source),
        "-o",
        str(output),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--output", type=Path, default=Path("/tmp/trainscale-kernel-bench"))
    parser.add_argument("--nvcc-flag", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    candidate = cuda_home / "bin" / "nvcc"
    nvcc = str(candidate) if candidate.exists() else shutil.which("nvcc")
    if nvcc is None:
        raise SystemExit("nvcc not found; set CUDA_HOME or PATH")
    source = Path(__file__).resolve().with_name("kernel_bench.cu")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        source=source,
        output=output,
        architecture=args.architecture,
        nvcc=nvcc,
        extra_flags=args.nvcc_flag,
    )
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
