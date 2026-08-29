from __future__ import annotations

import sys
from pathlib import Path

import torch

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_parallel.sharding import (  # noqa: E402
    balanced_shard_ranges,
    load_shard_checkpoint,
    reconstruct_tensor,
    save_shard_checkpoint,
    shard_tensor,
)


def test_ragged_shards_cover_tensor_exactly_once() -> None:
    tensor = torch.arange(35).view(5, 7)
    ranges = balanced_shard_ranges(tensor.numel(), 4)
    assert ranges[0].numel - ranges[-1].numel <= 1
    assert ranges[0].start == 0 and ranges[-1].stop == tensor.numel()
    shards = shard_tensor(tensor, 4)
    torch.testing.assert_close(reconstruct_tensor(shards, tuple(tensor.shape)), tensor)


def test_shard_checkpoint_manifest_restores_full_tensor(tmp_path: Path) -> None:
    tensor = torch.randn(3, 11)
    directory = tmp_path / "checkpoint"
    manifest = save_shard_checkpoint(directory, tensor, world_size=4)
    assert manifest["world_size"] == 4
    torch.testing.assert_close(load_shard_checkpoint(directory), tensor)
