"""Report whether the current interpreter can run the module 03 backends."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def collect() -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "schema_version": 1,
        "timestamp": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "distributed_available": dist.is_available(),
        "gloo_available": dist.is_gloo_available(),
        "nccl_available": dist.is_nccl_available(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_names": gpu_names,
        "driver": (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip()
            if torch.cuda.is_available()
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = collect()
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(args.output)
    required = payload["distributed_available"] and payload["gloo_available"]
    print(f"cpu_distributed_ready={required}")
    raise SystemExit(0 if required else 1)


if __name__ == "__main__":
    main()

