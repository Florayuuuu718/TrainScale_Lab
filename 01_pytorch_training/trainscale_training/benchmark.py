"""Controlled FP32, AMP, and torch.compile ablation harness."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from .config import load_config
from .engine import run_training
from .reporting import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = load_config(args.config)
    if base.device != "cuda":
        raise SystemExit("the FP32/AMP/compile ablation requires a CUDA config")

    root = Path(base.output_dir).parent
    variants = [
        ("fp32_eager", "fp32", False),
        ("amp_eager", "amp", False),
        ("fp32_compile", "fp32", True),
    ]
    results = []
    for name, precision, compile_model in variants:
        print(f"\n=== {name} ===")
        if compile_model:
            cache_parent = root / "torchinductor_cache"
            cache_parent.mkdir(parents=True, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(
                prefix="cold_", dir=cache_parent
            )
        config = base.with_overrides(
            experiment_name=name,
            precision=precision,
            compile_model=compile_model,
            output_dir=str(root / name),
            resume=None,
        )
        try:
            summary = run_training(config)
            history = summary["history"]
            steady_rows = history[1:] if len(history) > 1 else history
            result = {
                "variant": name,
                "status": "completed",
                "precision": precision,
                "compile_model": compile_model,
                "total_wall_seconds": summary["total_wall_seconds"],
                "peak_cuda_memory_bytes": summary["peak_cuda_memory_bytes"],
                "first_epoch_train_samples_per_second": history[0]["train"][
                    "samples_per_second"
                ],
                "steady_train_samples_per_second": sum(
                    row["train"]["samples_per_second"] for row in steady_rows
                )
                / len(steady_rows),
                "final_valid_loss": history[-1]["valid"]["loss"],
                "final_valid_accuracy": history[-1]["valid"]["accuracy"],
            }
        except Exception as error:  # experiment harness must persist failed variants
            result = {
                "variant": name,
                "status": "failed",
                "precision": precision,
                "compile_model": compile_model,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            print(f"{name} failed: {type(error).__name__}: {error}")
        results.append(result)
        write_json(args.output, {"base_config": base.to_dict(), "results": results})


if __name__ == "__main__":
    main()
