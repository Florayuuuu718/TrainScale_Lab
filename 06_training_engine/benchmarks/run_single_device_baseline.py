"""Prove the Tiny Transformer can overfit one fixed batch before reducer experiments."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
from torch import nn

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks"))

from artifact_contract import build_artifact  # noqa: E402
from trainscale_engine.contract import load_baseline_config  # noqa: E402
from trainscale_engine.model import (  # noqa: E402
    TinyTransformer,
    make_classification_batch,
    model_preset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_baseline_config(args.config)
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA baseline requested but CUDA is unavailable")
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    model_config = model_preset(config["model_preset"])
    model = TinyTransformer(model_config).to(device)
    tokens, labels = make_classification_batch(
        config["batch_size"], model_config, config["seed"] + 1
    )
    tokens, labels = tokens.to(device), labels.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    losses = []
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(config["steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(tokens), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    maximum_update = max(
        float((parameter.detach() - initial).abs().max())
        for parameter, initial in zip(model.parameters(), initial_parameters, strict=True)
    )
    passed = losses[-1] < losses[0] * 0.25 and maximum_update > 0
    payload = build_artifact(
        artifact_type="module06.single_device_baseline",
        repository_root=REPOSITORY_ROOT,
        environment={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device.type,
        },
        config=config,
        measurement={
            "workload": "repeat one fixed synthetic classification batch",
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        status="success" if passed else "failed",
        correctness={"status": "passed" if passed else "failed"},
        metrics={
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_ratio": losses[-1] / losses[0],
            "maximum_parameter_update": maximum_update,
            "duration_seconds": duration,
            "samples_per_second": config["batch_size"] * config["steps"] / duration,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "losses": losses,
        },
        raw_artifacts=[],
        boundary="Fixed-batch overfit proves trainability, not generalization or scaling.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
