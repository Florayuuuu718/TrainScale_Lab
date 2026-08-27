"""Calculate the module 03 DDP payload before renting multi-GPU hardware."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_nccl.contract import (  # noqa: E402
    load_bridge_config,
    module03_mlp_parameter_count,
)


def plan(config: dict[str, Any]) -> dict[str, object]:
    parameter_count = module03_mlp_parameter_count(
        int(config["input_dim"]),
        int(config["hidden_dim"]),
        int(config["num_classes"]),
    )
    fp32_bytes = parameter_count * 4
    bucket_cap_bytes = round(float(config["bucket_cap_mb"]) * 1024**2)
    return {
        "model": "module03 Linear-ReLU-Linear",
        "parameter_count": parameter_count,
        "fp32_gradient_payload_bytes": fp32_bytes,
        "fp32_gradient_payload_mib": fp32_bytes / 1024**2,
        "configured_bucket_cap_bytes": bucket_cap_bytes,
        "payload_to_bucket_cap_ratio": fp32_bytes / bucket_cap_bytes,
        "size_sweep_bracket_bytes": [
            2 ** math.floor(math.log2(fp32_bytes)),
            2 ** math.ceil(math.log2(fp32_bytes)),
        ],
        "note": (
            "Payload is derived from model parameters and dtype. Actual DDP bucket readiness "
            "and collective timing must be verified by the multi-GPU timeline."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_bridge_config(args.config)
    payload = {"config": config, "payload": plan(config)}
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
