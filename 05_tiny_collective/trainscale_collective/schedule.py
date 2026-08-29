"""Pure schedule and communication-volume model for TinyCollective."""

from __future__ import annotations

from dataclasses import asdict, dataclass

PHASE_TAG_BASE = {
    "ring_reduce_scatter": 1000,
    "ring_all_gather": 2000,
    "central_reduce": 3000,
    "central_broadcast": 4000,
}


@dataclass(frozen=True)
class Transfer:
    phase: str
    step: int
    rank: int
    send_to: int
    recv_from: int
    send_chunk: int
    recv_chunk: int
    send_tag: int
    recv_tag: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def chunk_sizes(element_count: int, world_size: int) -> list[int]:
    if element_count < 0 or world_size < 2:
        raise ValueError("element_count must be non-negative and world_size must be >= 2")
    quotient, remainder = divmod(element_count, world_size)
    return [quotient + (index < remainder) for index in range(world_size)]


def message_tag(phase: str, step: int, chunk: int, world_size: int) -> int:
    if phase not in PHASE_TAG_BASE:
        raise ValueError(f"unsupported phase: {phase}")
    if not 0 <= step < world_size or not 0 <= chunk < world_size:
        raise ValueError("step and chunk must be valid world-size indices")
    return PHASE_TAG_BASE[phase] + step * world_size + chunk


def ring_reduce_scatter_schedule(rank: int, world_size: int) -> list[Transfer]:
    if not 0 <= rank < world_size or world_size < 2:
        raise ValueError("rank/world_size are invalid")
    transfers = []
    for step in range(world_size - 1):
        send_chunk = (rank - step) % world_size
        recv_chunk = (rank - step - 1) % world_size
        transfers.append(
            Transfer(
                phase="ring_reduce_scatter",
                step=step,
                rank=rank,
                send_to=(rank + 1) % world_size,
                recv_from=(rank - 1) % world_size,
                send_chunk=send_chunk,
                recv_chunk=recv_chunk,
                send_tag=message_tag("ring_reduce_scatter", step, send_chunk, world_size),
                recv_tag=message_tag("ring_reduce_scatter", step, recv_chunk, world_size),
            )
        )
    return transfers


def ring_owner(rank: int, world_size: int) -> int:
    return (rank + 1) % world_size


def ring_all_gather_schedule(rank: int, world_size: int) -> list[Transfer]:
    if not 0 <= rank < world_size or world_size < 2:
        raise ValueError("rank/world_size are invalid")
    transfers = []
    for step in range(world_size - 1):
        send_chunk = (rank - step + 1) % world_size
        recv_chunk = (rank - step) % world_size
        transfers.append(
            Transfer(
                phase="ring_all_gather",
                step=step,
                rank=rank,
                send_to=(rank + 1) % world_size,
                recv_from=(rank - 1) % world_size,
                send_chunk=send_chunk,
                recv_chunk=recv_chunk,
                send_tag=message_tag("ring_all_gather", step, send_chunk, world_size),
                recv_tag=message_tag("ring_all_gather", step, recv_chunk, world_size),
            )
        )
    return transfers


def ring_volume(element_count: int, world_size: int) -> dict[str, int]:
    sizes = chunk_sizes(element_count, world_size)
    schedules = [
        [
            *ring_reduce_scatter_schedule(rank, world_size),
            *ring_all_gather_schedule(rank, world_size),
        ]
        for rank in range(world_size)
    ]
    sent_per_rank = [sum(sizes[item.send_chunk] for item in schedule) for schedule in schedules]
    return {
        "rounds": 2 * (world_size - 1),
        "total_sent_elements": sum(sent_per_rank),
        "minimum_rank_sent_elements": min(sent_per_rank),
        "maximum_rank_sent_elements": max(sent_per_rank),
        "expected_total_sent_elements": 2 * (world_size - 1) * element_count,
    }
