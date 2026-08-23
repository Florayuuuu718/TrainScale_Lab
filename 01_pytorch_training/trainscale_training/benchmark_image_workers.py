"""Benchmark DataLoader workers with real JPEG decoding and steady-state epochs."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image, write_jpeg  # type: ignore[import-untyped]
from torchvision.transforms import v2  # type: ignore[import-untyped]

from .data import seed_worker


class RepeatedJpegDataset(Dataset[tuple[torch.Tensor, int]]):
    """Decode a small on-disk JPEG corpus repeatedly to create a long epoch."""

    def __init__(self, root: Path, samples: int, image_size: int) -> None:
        self.paths = sorted(root.glob("*.jpg"))
        if not self.paths:
            raise ValueError(f"no .jpg files found in {root}")
        if samples <= 0:
            raise ValueError("samples must be positive")
        self.samples = samples
        self.transform = v2.Compose(
            [
                v2.RandomResizedCrop((image_size, image_size), scale=(0.6, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path_index = index % len(self.paths)
        image = decode_image(str(self.paths[path_index]), mode=ImageReadMode.RGB)
        return self.transform(image), path_index


def prepare_jpegs(root: Path, count: int, source_size: int, seed: int) -> None:
    """Create a deterministic local JPEG corpus when one does not already exist."""
    if count <= 0 or source_size <= 0:
        raise ValueError("image count and source size must be positive")
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    for index in range(count):
        path = root / f"sample_{index:05d}.jpg"
        image = torch.randint(
            0, 256, (3, source_size, source_size), dtype=torch.uint8, generator=generator
        )
        write_jpeg(image, str(path), quality=90)


def iterate_epoch(loader: DataLoader[tuple[torch.Tensor, int]]) -> tuple[float, float]:
    started = time.perf_counter()
    seen = sum(images.shape[0] for images, _ in loader)
    seconds = time.perf_counter() - started
    return seconds, seen / seconds


def measure(num_workers: int, args: argparse.Namespace) -> dict[str, object]:
    dataset = RepeatedJpegDataset(args.image_root, args.samples, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
    )
    first_seconds, first_rate = iterate_epoch(loader)
    for _ in range(args.warmup_epochs - 1):
        iterate_epoch(loader)
    timed = [iterate_epoch(loader) for _ in range(args.timed_epochs)]
    seconds = [value[0] for value in timed]
    rates = [value[1] for value in timed]
    return {
        "num_workers": num_workers,
        "first_epoch_seconds": first_seconds,
        "first_epoch_samples_per_second": first_rate,
        "steady_epoch_seconds": seconds,
        "median_steady_samples_per_second": statistics.median(rates),
        "min_steady_samples_per_second": min(rates),
        "max_steady_samples_per_second": max(rates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan DataLoader workers using on-disk JPEG decode and long epochs."
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--prepare-images", type=int, default=0)
    parser.add_argument("--source-size", type=int, default=160)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--timed-epochs", type=int, default=3)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup_epochs < 1 or args.timed_epochs < 1:
        parser.error("--warmup-epochs and --timed-epochs must be at least 1")
    if args.prepare_images:
        prepare_jpegs(args.image_root, args.prepare_images, args.source_size, args.seed)
    value = {
        "experiment": "dataloader_image_workers_steady_state",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "controlled_variables": {
            "image_root": str(args.image_root),
            "jpeg_files": len(list(args.image_root.glob("*.jpg"))),
            "samples_per_epoch": args.samples,
            "batch_size": args.batch_size,
            "source_size": args.source_size,
            "image_size": args.image_size,
            "warmup_epochs": args.warmup_epochs,
            "timed_epochs": args.timed_epochs,
            "prefetch_factor": args.prefetch_factor,
            "seed": args.seed,
        },
        "results": [measure(workers, args) for workers in args.workers],
    }
    output = json.dumps(value, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
