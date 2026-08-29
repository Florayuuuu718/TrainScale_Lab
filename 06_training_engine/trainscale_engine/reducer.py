"""Bulk, per-parameter, and bucketed teaching gradient reducers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from .bucket import BucketSpec, bucket_plan_digest, build_bucket_plan


@dataclass
class ReducerStepStats:
    strategy: str
    synchronized: bool
    collective_count: int = 0
    payload_bytes: int = 0
    timeline: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ReducerState:
    def __init__(self, strategy: str) -> None:
        self.strategy = strategy
        self.active = False
        self.sync = False
        self.stats = ReducerStepStats(strategy=strategy, synchronized=False)

    def begin_backward(self, *, sync: bool) -> None:
        if self.active:
            raise RuntimeError("previous backward has not been finalized")
        self.active = True
        self.sync = sync
        self.stats = ReducerStepStats(strategy=self.strategy, synchronized=sync)
        self._event("backward_start")

    def _require_active(self) -> None:
        if not self.active:
            raise RuntimeError("begin_backward must be called first")

    def _event(self, kind: str, **details: Any) -> None:
        self.stats.timeline.append(
            {"kind": kind, "timestamp_ns": time.perf_counter_ns(), **details}
        )

    def assert_complete(self) -> None:
        if self.active:
            raise RuntimeError("optimizer step is forbidden before reducer finalization")


class BulkReducer(_ReducerState):
    """Flatten every available gradient once after backward."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__("bulk")
        self.parameters = tuple(model.parameters())

    def finish_backward(self) -> ReducerStepStats:
        self._require_active()
        self._event("backward_complete")
        if self.sync:
            gradients = [
                parameter.grad for parameter in self.parameters if parameter.grad is not None
            ]
            if gradients:
                flat = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
                self._event(
                    "collective_launch", bucket=0, payload_bytes=flat.numel() * flat.element_size()
                )
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat.div_(dist.get_world_size())
                self._event("collective_complete", bucket=0)
                offset = 0
                for gradient in gradients:
                    assert gradient is not None
                    count = gradient.numel()
                    gradient.copy_(flat[offset : offset + count].view_as(gradient))
                    offset += count
                self.stats.collective_count = 1
                self.stats.payload_bytes = flat.numel() * flat.element_size()
        self.active = False
        return self.stats


class PerParameterReducer(_ReducerState):
    """Synchronize each gradient from its autograd hook."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__("per_parameter")
        self.handles = []
        for name, parameter in model.named_parameters():
            self.handles.append(parameter.register_hook(self._hook(name, parameter)))

    def _hook(self, name: str, parameter: nn.Parameter) -> Any:
        def synchronize(gradient: torch.Tensor) -> torch.Tensor:
            self._require_active()
            self._event("gradient_ready", parameter=name)
            if not self.sync:
                return gradient
            combined = gradient.detach().clone()
            previous = parameter.grad
            if previous is not None:
                combined.add_(previous)
            payload_bytes = combined.numel() * combined.element_size()
            self._event("collective_launch", parameter=name, payload_bytes=payload_bytes)
            dist.all_reduce(combined, op=dist.ReduceOp.SUM)
            combined.div_(dist.get_world_size())
            self._event("collective_complete", parameter=name)
            self.stats.collective_count += 1
            self.stats.payload_bytes += payload_bytes
            return combined if previous is None else combined - previous

        return synchronize

    def finish_backward(self) -> ReducerStepStats:
        self._require_active()
        self._event("backward_complete")
        self.active = False
        return self.stats


class BucketReducer(_ReducerState):
    """Launch deterministic flat buckets when all of their gradients are ready."""

    def __init__(self, model: nn.Module, bucket_cap_bytes: int, *, asynchronous: bool) -> None:
        super().__init__("bucket_async" if asynchronous else "bucket_sync")
        self.model = model
        self.plan = build_bucket_plan(model, bucket_cap_bytes)
        self.asynchronous = asynchronous
        self.named_parameters = dict(model.named_parameters())
        self.parameter_to_bucket = {
            entry.name: bucket.index for bucket in self.plan for entry in bucket.entries
        }
        first = next(model.parameters())
        self.buffers = {
            bucket.index: torch.empty(bucket.numel, dtype=first.dtype, device=first.device)
            for bucket in self.plan
        }
        self.ready: set[str] = set()
        self.launched: set[int] = set()
        self.works: dict[int, Any] = {}
        self._validate_plan_across_ranks()
        self.handles = [
            parameter.register_hook(self._hook(name, parameter))
            for name, parameter in model.named_parameters()
        ]

    def _validate_plan_across_ranks(self) -> None:
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return
        digest = bucket_plan_digest(self.plan)
        gathered: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, digest)
        if any(item != digest for item in gathered):
            raise RuntimeError(f"bucket plan differs across ranks: {gathered}")

    def begin_backward(self, *, sync: bool) -> None:
        super().begin_backward(sync=sync)
        self.ready.clear()
        self.launched.clear()
        self.works.clear()

    def _entry(self, bucket: BucketSpec, name: str) -> Any:
        return next(entry for entry in bucket.entries if entry.name == name)

    def _hook(self, name: str, parameter: nn.Parameter) -> Any:
        def mark_ready(gradient: torch.Tensor) -> torch.Tensor:
            self._require_active()
            self._event("gradient_ready", parameter=name, bucket=self.parameter_to_bucket[name])
            if not self.sync:
                return gradient
            bucket_index = self.parameter_to_bucket[name]
            bucket = self.plan[bucket_index]
            entry = self._entry(bucket, name)
            combined = gradient.detach()
            if parameter.grad is not None:
                combined = combined + parameter.grad
            self.buffers[bucket_index][entry.offset : entry.offset + entry.numel].copy_(
                combined.reshape(-1)
            )
            self.ready.add(name)
            if all(item.name in self.ready for item in bucket.entries):
                self._launch(bucket)
            return gradient

        return mark_ready

    def _launch(self, bucket: BucketSpec) -> None:
        if bucket.index in self.launched:
            raise RuntimeError(f"bucket {bucket.index} launched more than once")
        buffer = self.buffers[bucket.index]
        self._event("collective_launch", bucket=bucket.index, payload_bytes=bucket.bytes)
        work = dist.all_reduce(
            buffer,
            op=dist.ReduceOp.SUM,
            async_op=self.asynchronous,
        )
        self.launched.add(bucket.index)
        if self.asynchronous:
            self.works[bucket.index] = work
        else:
            self._event("collective_complete", bucket=bucket.index)
        self.stats.collective_count += 1
        self.stats.payload_bytes += bucket.bytes

    def _fill_unlaunched(self, bucket: BucketSpec) -> None:
        buffer = self.buffers[bucket.index]
        for entry in bucket.entries:
            if entry.name in self.ready:
                continue
            parameter = self.named_parameters[entry.name]
            target = buffer[entry.offset : entry.offset + entry.numel]
            if parameter.grad is None:
                target.zero_()
            else:
                target.copy_(parameter.grad.detach().reshape(-1))
                self.ready.add(entry.name)

    def finish_backward(self) -> ReducerStepStats:
        self._require_active()
        self._event("backward_complete")
        if self.sync:
            for bucket in self.plan:
                if bucket.index not in self.launched:
                    self._fill_unlaunched(bucket)
                    self._launch(bucket)
            for bucket in self.plan:
                if self.asynchronous:
                    self.works[bucket.index].wait()
                    self._event("collective_complete", bucket=bucket.index)
                buffer = self.buffers[bucket.index]
                buffer.div_(dist.get_world_size())
                for entry in bucket.entries:
                    parameter = self.named_parameters[entry.name]
                    if parameter.grad is not None:
                        parameter.grad.copy_(
                            buffer[entry.offset : entry.offset + entry.numel].view(entry.shape)
                        )
        self.active = False
        return self.stats


__all__ = [
    "BulkReducer",
    "BucketReducer",
    "PerParameterReducer",
    "ReducerStepStats",
    "build_bucket_plan",
]
