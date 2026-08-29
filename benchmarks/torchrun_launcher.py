"""Persistent single-node torchrun launcher shared by Modules 06 and 07."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

RANK_RESULT_PATTERN = re.compile(r"rank_\d+\.json")


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def torchrun_command(
    worker: Path, world_size: int, rank_directory: Path, worker_args: list[str]
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
        str(worker),
        "--rank-directory",
        str(rank_directory),
        *worker_args,
    ]


def rank_result_files(rank_directory: Path) -> list[Path]:
    """Return worker result JSON files without mistaking profiler traces for ranks."""
    return sorted(
        path
        for path in rank_directory.glob("rank_*.json")
        if RANK_RESULT_PATTERN.fullmatch(path.name)
    )


def launch_torchrun(
    *,
    repository_root: Path,
    worker: Path,
    world_size: int,
    rank_directory: Path,
    worker_args: list[str],
    python_paths: list[Path],
    timeout_seconds: int,
) -> dict[str, Any]:
    rank_directory.mkdir(parents=True, exist_ok=False)
    command = torchrun_command(worker, world_size, rank_directory, worker_args)
    (rank_directory / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    environment = os.environ.copy()
    paths = [str(path) for path in python_paths]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            env=environment,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = None
        stdout = _text(error.stdout)
        stderr = _text(error.stderr)
    (rank_directory / "stdout.log").write_text(stdout, encoding="utf-8")
    (rank_directory / "stderr.log").write_text(stderr, encoding="utf-8")
    rank_files = rank_result_files(rank_directory)
    ranks = [json.loads(path.read_text(encoding="utf-8")) for path in rank_files]
    return {
        "status": "success"
        if not timed_out and returncode == 0 and len(ranks) == world_size
        else "failed",
        "world_size": world_size,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "ranks": ranks,
    }
