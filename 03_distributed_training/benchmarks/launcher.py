"""Launch isolated local torchrun jobs and collect one JSON result per rank."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
WORKER = MODULE_ROOT / "trainscale_distributed" / "worker.py"


def torchrun_command(world_size: int, rank_directory: Path, worker_args: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
        str(WORKER),
        "--rank-directory",
        str(rank_directory),
        *worker_args,
    ]


def launch(
    *,
    world_size: int,
    rank_directory: Path,
    worker_args: list[str],
    timeout_seconds: int,
) -> dict[str, Any]:
    rank_directory.mkdir(parents=True, exist_ok=False)
    command = torchrun_command(world_size, rank_directory, worker_args)
    environment = os.environ.copy()
    python_path = str(MODULE_ROOT)
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout_seconds,
    )
    rank_files = sorted(rank_directory.glob("rank_*.json"))
    ranks = [json.loads(path.read_text(encoding="utf-8")) for path in rank_files]
    return {
        "status": "success" if completed.returncode == 0 and len(ranks) == world_size else "failed",
        "world_size": world_size,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "ranks": ranks,
    }
