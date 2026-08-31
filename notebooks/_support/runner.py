"""Safely launch existing project runners outside the Jupyter kernel."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    elapsed_seconds: float
    log_path: Path
    stdout_tail: str
    stderr_tail: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join(text.splitlines()[-lines:])


def run_command(
    args: Sequence[str],
    *,
    cwd: Path,
    output_dir: Path,
    label: str,
    timeout_seconds: int = 600,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    """Run an argv list, preserve a log, and raise after recording failures."""
    if not args or any(not isinstance(item, str) or not item for item in args):
        raise ValueError("命令必须是非空字符串组成的参数列表。")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{label}.log"
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            env=child_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode("utf-8", "replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", "replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        stderr += f"\nTimed out after {timeout_seconds} seconds."
        returncode = 124
    elapsed = time.monotonic() - started
    command_line = subprocess.list2cmdline(list(args))
    log_path.write_text(
        f"command: {command_line}\nreturncode: {returncode}\nelapsed_seconds: {elapsed:.3f}\n"
        f"\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        encoding="utf-8",
    )
    result = CommandResult(
        args=tuple(args),
        returncode=returncode,
        elapsed_seconds=elapsed,
        log_path=log_path,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
    )
    if not result.passed:
        raise RuntimeError(
            f"命令失败（exit={returncode}）。日志保留在 {log_path.relative_to(cwd)}\n"
            f"stderr 尾部：\n{result.stderr_tail}"
        )
    return result
