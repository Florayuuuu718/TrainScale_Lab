"""Pure shard ownership used to validate FSDP state and checkpoint semantics offline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ShardRange:
    rank: int
    start: int
    stop: int

    @property
    def numel(self) -> int:
        return self.stop - self.start


def balanced_shard_ranges(element_count: int, world_size: int) -> tuple[ShardRange, ...]:
    if element_count < 0 or world_size <= 0:
        raise ValueError("element_count must be non-negative and world_size positive")
    quotient, remainder = divmod(element_count, world_size)
    ranges = []
    start = 0
    for rank in range(world_size):
        count = quotient + (rank < remainder)
        ranges.append(ShardRange(rank=rank, start=start, stop=start + count))
        start += count
    return tuple(ranges)


def shard_tensor(tensor: torch.Tensor, world_size: int) -> tuple[torch.Tensor, ...]:
    flattened = tensor.detach().contiguous().view(-1)
    return tuple(
        flattened[item.start : item.stop].clone()
        for item in balanced_shard_ranges(flattened.numel(), world_size)
    )


def reconstruct_tensor(
    shards: tuple[torch.Tensor, ...] | list[torch.Tensor], shape: tuple[int, ...]
) -> torch.Tensor:
    expected = 1
    for dimension in shape:
        expected *= dimension
    flattened = torch.cat(list(shards)) if shards else torch.empty(0)
    if flattened.numel() != expected:
        raise ValueError("shards do not match the requested shape")
    return flattened.view(shape)


def save_shard_checkpoint(
    directory: Path, tensor: torch.Tensor, *, world_size: int
) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=False)
    shards = shard_tensor(tensor, world_size)
    ranges = balanced_shard_ranges(tensor.numel(), world_size)
    for item, shard in zip(ranges, shards, strict=True):
        torch.save(shard, directory / f"rank_{item.rank}.pt")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "world_size": world_size,
        "ranges": [asdict(item) for item in ranges],
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_shard_checkpoint(directory: Path) -> torch.Tensor:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported shard checkpoint schema")
    world_size = int(manifest["world_size"])
    shards = [
        torch.load(directory / f"rank_{rank}.pt", map_location="cpu", weights_only=True)
        for rank in range(world_size)
    ]
    return reconstruct_tensor(shards, tuple(manifest["shape"]))
