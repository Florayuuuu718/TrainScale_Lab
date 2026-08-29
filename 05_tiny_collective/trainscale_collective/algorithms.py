"""Minimal centralized and ring AllReduce implementations using distributed P2P."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from .schedule import (
    Transfer,
    chunk_sizes,
    message_tag,
    ring_all_gather_schedule,
    ring_owner,
    ring_reduce_scatter_schedule,
)


@dataclass
class PendingExchange:
    requests: list[Any]
    received: torch.Tensor
    completed: bool = False

    def wait(self) -> torch.Tensor:
        if self.completed:
            raise RuntimeError("an asynchronous P2P exchange may only be waited once")
        for request in self.requests:
            request.wait()
        self.completed = True
        return self.received


def launch_exchange(
    send_tensor: torch.Tensor,
    *,
    recv_elements: int,
    send_to: int,
    recv_from: int,
    send_tag: int,
    recv_tag: int,
) -> PendingExchange:
    received = torch.empty(recv_elements, dtype=send_tensor.dtype, device=send_tensor.device)
    operations = [
        dist.P2POp(dist.isend, send_tensor.contiguous(), send_to, tag=send_tag),
        dist.P2POp(dist.irecv, received, recv_from, tag=recv_tag),
    ]
    return PendingExchange(dist.batch_isend_irecv(operations), received)


def _trace(transfer: Transfer, sizes: list[int]) -> dict[str, int | str]:
    return {
        **transfer.to_dict(),
        "send_elements": sizes[transfer.send_chunk],
        "recv_elements": sizes[transfer.recv_chunk],
    }


def centralized_all_reduce(
    tensor: torch.Tensor, root: int = 0
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    rank, world_size = dist.get_rank(), dist.get_world_size()
    flattened = tensor.contiguous().view(-1)
    result = flattened.clone()
    trace: list[dict[str, Any]] = []
    if rank == root:
        for peer in range(world_size):
            if peer == root:
                continue
            received = torch.empty_like(result)
            tag = message_tag("central_reduce", peer, peer, world_size)
            dist.recv(received, src=peer, tag=tag)
            result.add_(received)
            trace.append(
                {"phase": "central_reduce", "peer": peer, "tag": tag, "elements": result.numel()}
            )
        for peer in range(world_size):
            if peer == root:
                continue
            tag = message_tag("central_broadcast", peer, peer, world_size)
            dist.send(result, dst=peer, tag=tag)
            trace.append(
                {"phase": "central_broadcast", "peer": peer, "tag": tag, "elements": result.numel()}
            )
    else:
        reduce_tag = message_tag("central_reduce", rank, rank, world_size)
        dist.send(result, dst=root, tag=reduce_tag)
        broadcast_tag = message_tag("central_broadcast", rank, rank, world_size)
        dist.recv(result, src=root, tag=broadcast_tag)
        trace.extend(
            [
                {
                    "phase": "central_reduce",
                    "peer": root,
                    "tag": reduce_tag,
                    "elements": result.numel(),
                },
                {
                    "phase": "central_broadcast",
                    "peer": root,
                    "tag": broadcast_tag,
                    "elements": result.numel(),
                },
            ]
        )
    return result.view_as(tensor), trace


def ring_reduce_scatter(tensor: torch.Tensor) -> tuple[torch.Tensor, int, list[dict[str, Any]]]:
    rank, world_size = dist.get_rank(), dist.get_world_size()
    sizes = chunk_sizes(tensor.numel(), world_size)
    chunks = [chunk.clone() for chunk in torch.split(tensor.contiguous().view(-1), sizes)]
    trace: list[dict[str, Any]] = []
    for transfer in ring_reduce_scatter_schedule(rank, world_size):
        pending = launch_exchange(
            chunks[transfer.send_chunk],
            recv_elements=sizes[transfer.recv_chunk],
            send_to=transfer.send_to,
            recv_from=transfer.recv_from,
            send_tag=transfer.send_tag,
            recv_tag=transfer.recv_tag,
        )
        chunks[transfer.recv_chunk].add_(pending.wait())
        trace.append(_trace(transfer, sizes))
    owner = ring_owner(rank, world_size)
    return chunks[owner], owner, trace


def ring_all_gather(
    local_chunk: torch.Tensor, chunk_sizes_: list[int], owner: int
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if len(chunk_sizes_) != world_size or owner != ring_owner(rank, world_size):
        raise ValueError("chunk sizes or owner do not match the ring schedule")
    chunks = [
        torch.empty(size, dtype=local_chunk.dtype, device=local_chunk.device)
        for size in chunk_sizes_
    ]
    chunks[owner].copy_(local_chunk)
    trace: list[dict[str, Any]] = []
    for transfer in ring_all_gather_schedule(rank, world_size):
        pending = launch_exchange(
            chunks[transfer.send_chunk],
            recv_elements=chunk_sizes_[transfer.recv_chunk],
            send_to=transfer.send_to,
            recv_from=transfer.recv_from,
            send_tag=transfer.send_tag,
            recv_tag=transfer.recv_tag,
        )
        chunks[transfer.recv_chunk].copy_(pending.wait())
        trace.append(_trace(transfer, chunk_sizes_))
    return torch.cat(chunks), trace


def ring_all_reduce(tensor: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    reduced, owner, reduce_trace = ring_reduce_scatter(tensor)
    gathered, gather_trace = ring_all_gather(
        reduced, chunk_sizes(tensor.numel(), dist.get_world_size()), owner
    )
    return gathered.view_as(tensor), [*reduce_trace, *gather_trace]
