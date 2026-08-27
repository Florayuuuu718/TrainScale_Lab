"""Environment, GPU capability, and topology probes for module 04."""

from __future__ import annotations

import platform
import shutil
import subprocess
from datetime import datetime
from typing import Any


def _command_output(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"status": "unavailable", "command": command, "reason": "executable not found"}
    completed = subprocess.run(command, capture_output=True, check=False, text=True)
    return {
        "status": "success" if completed.returncode == 0 else "failed",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def collect_environment() -> dict[str, Any]:
    """Collect facts without treating missing multi-GPU hardware as a code failure."""
    import torch
    import torch.distributed as dist

    nccl_version = torch.cuda.nccl.version() if dist.is_nccl_available() else None
    if isinstance(nccl_version, tuple):
        nccl_version = list(nccl_version)
    device_count = torch.cuda.device_count()
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": device_count,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(device_count)],
        "gpu_compute_capabilities": [
            list(torch.cuda.get_device_capability(index)) for index in range(device_count)
        ],
        "nccl_available": dist.is_nccl_available(),
        "nccl_version": nccl_version,
        "driver_query": _command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "gpu_list": _command_output(["nvidia-smi", "-L"]),
        "topology": _command_output(["nvidia-smi", "topo", "-m"]),
        "module04_capability": {
            "linux_required": platform.system() == "Linux",
            "minimum_gpu_count": 2,
            "multi_gpu_ready": (
                platform.system() == "Linux"
                and dist.is_nccl_available()
                and torch.cuda.is_available()
                and device_count >= 2
            ),
        },
    }

