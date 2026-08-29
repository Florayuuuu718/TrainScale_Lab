"""Deterministic gradient bucket ownership and flat-buffer offsets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from torch import nn


@dataclass(frozen=True)
class BucketEntry:
    name: str
    parameter_index: int
    offset: int
    numel: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class BucketSpec:
    index: int
    numel: int
    bytes: int
    entries: tuple[BucketEntry, ...]


def build_bucket_plan(model: nn.Module, bucket_cap_bytes: int) -> tuple[BucketSpec, ...]:
    """Pack reverse parameter order, matching the usual backward-ready direction."""
    if bucket_cap_bytes <= 0:
        raise ValueError("bucket_cap_bytes must be positive")
    named = [
        (index, name, parameter) for index, (name, parameter) in enumerate(model.named_parameters())
    ]
    if not named:
        raise ValueError("model has no parameters")
    dtypes = {parameter.dtype for _, _, parameter in named}
    devices = {parameter.device for _, _, parameter in named}
    if len(dtypes) != 1 or len(devices) != 1:
        raise ValueError("the teaching reducer requires one parameter dtype and device")
    element_size = named[0][2].element_size()
    buckets: list[BucketSpec] = []
    pending: list[tuple[int, str, nn.Parameter]] = []
    pending_bytes = 0

    def flush() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        entries = []
        offset = 0
        for parameter_index, name, parameter in pending:
            entries.append(
                BucketEntry(
                    name=name,
                    parameter_index=parameter_index,
                    offset=offset,
                    numel=parameter.numel(),
                    shape=tuple(parameter.shape),
                )
            )
            offset += parameter.numel()
        buckets.append(
            BucketSpec(
                index=len(buckets),
                numel=offset,
                bytes=offset * element_size,
                entries=tuple(entries),
            )
        )
        pending = []
        pending_bytes = 0

    for parameter_index, name, parameter in reversed(named):
        parameter_bytes = parameter.numel() * parameter.element_size()
        if pending and pending_bytes + parameter_bytes > bucket_cap_bytes:
            flush()
        pending.append((parameter_index, name, parameter))
        pending_bytes += parameter_bytes
        if parameter_bytes >= bucket_cap_bytes:
            flush()
    flush()
    return tuple(buckets)


def bucket_plan_digest(plan: tuple[BucketSpec, ...]) -> str:
    payload = [asdict(bucket) for bucket in plan]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
