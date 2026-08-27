"""Inspect whether this interpreter and host can run module 04 experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_nccl.environment import collect_environment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-multi-gpu", action="store_true", help="fail unless Linux/NCCL and 2 GPUs exist"
    )
    args = parser.parse_args()
    payload = collect_environment()
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output)
    ready = payload["module04_capability"]["multi_gpu_ready"]
    print(f"module04_multi_gpu_ready={ready}")
    raise SystemExit(1 if args.require_multi_gpu and not ready else 0)


if __name__ == "__main__":
    main()

