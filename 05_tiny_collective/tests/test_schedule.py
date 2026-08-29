from __future__ import annotations

import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_collective.schedule import (  # noqa: E402
    chunk_sizes,
    ring_all_gather_schedule,
    ring_owner,
    ring_reduce_scatter_schedule,
    ring_volume,
)


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_ring_schedule_neighbors_rounds_and_owners(world_size: int) -> None:
    for rank in range(world_size):
        reduce = ring_reduce_scatter_schedule(rank, world_size)
        gather = ring_all_gather_schedule(rank, world_size)
        assert len(reduce) == len(gather) == world_size - 1
        assert all(item.send_to == (rank + 1) % world_size for item in [*reduce, *gather])
        assert all(item.recv_from == (rank - 1) % world_size for item in [*reduce, *gather])
        assert reduce[-1].recv_chunk == ring_owner(rank, world_size)
        assert gather[0].send_chunk == ring_owner(rank, world_size)
        next_reduce = ring_reduce_scatter_schedule((rank + 1) % world_size, world_size)
        next_gather = ring_all_gather_schedule((rank + 1) % world_size, world_size)
        for sent, received in zip(reduce, next_reduce, strict=True):
            assert sent.send_chunk == received.recv_chunk
            assert sent.send_tag == received.recv_tag
        for sent, received in zip(gather, next_gather, strict=True):
            assert sent.send_chunk == received.recv_chunk
            assert sent.send_tag == received.recv_tag


def test_ragged_chunks_preserve_every_element() -> None:
    assert chunk_sizes(17, 3) == [6, 6, 5]
    assert chunk_sizes(5, 4) == [2, 1, 1, 1]
    assert sum(chunk_sizes(17, 4)) == 17


@pytest.mark.parametrize(("elements", "world_size"), [(17, 3), (17, 4), (16, 4)])
def test_ring_volume_matches_closed_form(elements: int, world_size: int) -> None:
    volume = ring_volume(elements, world_size)
    assert volume["rounds"] == 2 * (world_size - 1)
    assert volume["total_sent_elements"] == 2 * (world_size - 1) * elements
    assert volume["total_sent_elements"] == volume["expected_total_sent_elements"]
