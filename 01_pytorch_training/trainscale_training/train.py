"""CLI for a configured single-device training experiment."""

from __future__ import annotations

import argparse

from .config import load_config
from .engine import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a module 01 TOML config")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--precision", choices=("fp32", "amp"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config).with_overrides(
        device=args.device,
        precision=args.precision,
        epochs=args.epochs,
        resume=args.resume,
        output_dir=args.output_dir,
        compile_model=args.compile,
    )
    run_training(config)


if __name__ == "__main__":
    main()
