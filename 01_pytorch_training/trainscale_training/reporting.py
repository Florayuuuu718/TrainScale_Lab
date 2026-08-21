"""Small dependency-free result and environment writers."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

import psutil
import torch


def environment_record() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_count": torch.cuda.device_count(),
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "memory_total_bytes": psutil.virtual_memory().total,
    }


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
