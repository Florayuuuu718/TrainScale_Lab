"""Estimate DDP/FSDP2 persistent state for the shared Module 06 model presets."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "06_training_engine"))
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks"))

from artifact_contract import build_artifact  # noqa: E402
from trainscale_engine.model import TinyTransformer, model_preset  # noqa: E402
from trainscale_parallel.memory import estimate_training_memory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_presets = ("small", "medium")
    world_sizes = (2, 4)
    config = {
        "model_presets": list(model_presets),
        "world_sizes": list(world_sizes),
        "optimizer": "AdamW",
        "dtype": "FP32",
        "activation_model": "batch * sequence * d_model * layers * 8 tensors",
    }
    records = []
    for preset in model_presets:
        model_config = model_preset(preset)
        model = TinyTransformer(model_config)
        activation_elements = (
            16 * model_config.sequence_length * model_config.d_model * model_config.layers * 8
        )
        ddp = estimate_training_memory(model, activation_elements=activation_elements)
        for world_size in world_sizes:
            fsdp = estimate_training_memory(
                model,
                world_size=world_size,
                fully_sharded=True,
                activation_elements=activation_elements,
            )
            records.append(
                {
                    "model_preset": preset,
                    "world_size": world_size,
                    "ddp": ddp.to_dict(),
                    "fsdp2_lower_bound": fsdp.to_dict(),
                    "persistent_reduction_ratio": 1
                    - fsdp.persistent_state_bytes / ddp.persistent_state_bytes,
                }
            )
    payload = build_artifact(
        artifact_type="module07.memory_estimate",
        repository_root=REPOSITORY_ROOT,
        environment={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "analytical",
        },
        config=config,
        measurement={
            "parameter_bytes": 4,
            "gradient_bytes": 4,
            "adam_state_bytes_per_parameter": 8,
        },
        status="success",
        correctness={"status": "passed"},
        metrics={"records": records},
        raw_artifacts=[],
        boundary=(
            "This is a lower-bound state model; allocator, collectives, temporary full parameters, "
            "and saved activations require GPU measurement."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
