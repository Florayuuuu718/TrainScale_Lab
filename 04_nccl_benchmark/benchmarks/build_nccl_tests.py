"""Print or execute the pinned nccl-tests clone and build commands."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

NCCL_TESTS_REPOSITORY = "https://github.com/NVIDIA/nccl-tests.git"
NCCL_TESTS_TAG = "v2.19.7"
NCCL_TESTS_COMMIT = "1a65d7f0514b8da6a61ae235d1c5f38549478e29"


def clone_command(source_directory: Path) -> list[str]:
    return [
        "git",
        "clone",
        "--branch",
        NCCL_TESTS_TAG,
        "--depth",
        "1",
        NCCL_TESTS_REPOSITORY,
        str(source_directory),
    ]


def build_command(source_directory: Path, jobs: int) -> list[str]:
    return ["make", "-C", str(source_directory), f"-j{jobs}"]


def resolved_commit(source_directory: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_directory, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument(
        "--nvcc-flag",
        action="append",
        default=[],
        help="repeatable flag passed through NVIDIA's NVCC_PREPEND_FLAGS environment variable",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    clone = clone_command(args.source_directory)
    build = build_command(args.source_directory, args.jobs)
    print("pinned_tag=", NCCL_TESTS_TAG)
    print("pinned_commit=", NCCL_TESTS_COMMIT)
    print("clone:", subprocess.list2cmdline(clone))
    print("build:", subprocess.list2cmdline(build))
    print("NVCC_PREPEND_FLAGS=", " ".join(args.nvcc_flag))
    if not args.execute:
        print("dry_run=true; pass --execute on the Linux GPU host")
        return
    if os.name == "nt":
        raise SystemExit("nccl-tests build requires Linux/WSL, not native Windows")
    if not args.source_directory.exists():
        subprocess.run(clone, check=True)
    actual = resolved_commit(args.source_directory)
    if actual != NCCL_TESTS_COMMIT:
        raise SystemExit(f"unexpected nccl-tests commit: {actual}")
    environment = os.environ.copy()
    if args.nvcc_flag:
        environment["NVCC_PREPEND_FLAGS"] = " ".join(args.nvcc_flag)
    try:
        subprocess.run(build, check=True, env=environment)
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            "nccl-tests build failed; keep the compiler output as environment evidence "
            f"(returncode={error.returncode})"
        ) from None
    print(f"binary_directory={args.source_directory / 'build'}")


if __name__ == "__main__":
    main()
