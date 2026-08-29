"""Launch one isolated TinyCollective torchrun job and collect rank artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
WORKER = MODULE_ROOT / "trainscale_collective" / "worker.py"


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


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
    *, world_size: int, rank_directory: Path, worker_args: list[str], timeout_seconds: int
) -> dict[str, Any]:
    rank_directory.mkdir(parents=True, exist_ok=False)
    command = torchrun_command(world_size, rank_directory, worker_args)
    (rank_directory / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(MODULE_ROOT) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
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
        stdout = _timeout_text(error.stdout)
        stderr = _timeout_text(error.stderr)
    (rank_directory / "stdout.log").write_text(stdout, encoding="utf-8")
    (rank_directory / "stderr.log").write_text(stderr, encoding="utf-8")
    rank_files = sorted(rank_directory.glob("rank_*.json"))
    return {
        "status": "success"
        if not timed_out and returncode == 0 and len(rank_files) == world_size
        else "failed",
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "ranks": [json.loads(path.read_text(encoding="utf-8")) for path in rank_files],
    }
