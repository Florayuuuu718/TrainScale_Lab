"""Locate the repository and describe a notebook run without assuming CUDA."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

NotebookMode = Literal["reference", "local", "gpu"]
VALID_MODES = frozenset({"reference", "local", "gpu"})


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upwards until the TrainScale ``pyproject.toml`` is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        project = candidate / "pyproject.toml"
        if project.is_file() and "trainscale-lab" in project.read_text(encoding="utf-8"):
            return candidate
    raise FileNotFoundError("找不到 TrainScale Lab 根目录；请从仓库内启动 Jupyter。")


def resolve_mode(requested: str = "reference") -> NotebookMode:
    """Resolve the editable cell value, allowing CI to force reference mode."""
    value = os.environ.get("TRAINSCALE_NOTEBOOK_MODE", requested).strip().lower()
    if value not in VALID_MODES:
        choices = ", ".join(sorted(VALID_MODES))
        raise ValueError(f"未知 Notebook 模式 {value!r}；可选值：{choices}")
    return cast(NotebookMode, value)


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _torch_status() -> tuple[str, int]:
    try:
        import torch
    except ImportError:
        return "not installed", 0
    try:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except RuntimeError:
        gpu_count = 0
    return str(torch.__version__), gpu_count


@dataclass(frozen=True)
class NotebookContext:
    repo_root: Path
    notebook: str
    mode: NotebookMode
    run_id: str
    output_dir: Path
    python: str
    platform: str
    torch: str
    gpu_count: int
    git_commit: str

    def card(self) -> dict[str, object]:
        """Return a display-safe status card (no username or absolute rental path)."""
        values = asdict(self)
        values["repo_root"] = self.repo_root.name
        values["output_dir"] = self.output_dir.relative_to(self.repo_root).as_posix()
        return values


def create_context(notebook: str, requested_mode: str = "reference") -> NotebookContext:
    root = find_repo_root()
    mode = resolve_mode(requested_mode)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "notebooks" / "_runs" / notebook / run_id
    output.mkdir(parents=True, exist_ok=False)
    torch_version, gpu_count = _torch_status()
    return NotebookContext(
        repo_root=root,
        notebook=notebook,
        mode=mode,
        run_id=run_id,
        output_dir=output,
        python=sys.version.split()[0],
        platform=platform.system(),
        torch=torch_version,
        gpu_count=gpu_count,
        git_commit=_git_commit(root),
    )
