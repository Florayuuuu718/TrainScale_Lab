"""Minimal train-step state machine that exposes reducer synchronization boundaries."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .reducer import BucketReducer, BulkReducer, PerParameterReducer, ReducerStepStats

ManualReducer = BulkReducer | PerParameterReducer | BucketReducer


@dataclass(frozen=True)
class StepResult:
    loss: float
    gradient_norm: float | None
    optimizer_step_skipped: bool
    collective_count: int
    payload_bytes: int
    timeline: tuple[dict[str, Any], ...]


def _autocast(device: torch.device, precision: str) -> Any:
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision != "amp":
        raise ValueError("precision must be fp32 or amp")
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    *,
    reducer: ManualReducer | None,
    accumulation_steps: int = 1,
    precision: str = "fp32",
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float | None = None,
    before_unscale: Callable[[nn.Module], None] | None = None,
    before_optimizer_step: Callable[[nn.Module], None] | None = None,
) -> StepResult:
    """Run one optimizer step; DDP is represented by reducer=None and a DDP model."""
    if accumulation_steps <= 0 or tokens.shape[0] % accumulation_steps:
        raise ValueError("local batch must be divisible by accumulation_steps")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    if reducer is None and not isinstance(model, DDP):
        raise ValueError("reducer=None is reserved for a DistributedDataParallel model")
    optimizer.zero_grad(set_to_none=True)
    token_chunks = tokens.chunk(accumulation_steps)
    label_chunks = labels.chunk(accumulation_steps)
    total_loss = 0.0
    step_stats: list[ReducerStepStats] = []
    use_scaler = scaler is not None and scaler.is_enabled()
    for index, (micro_tokens, micro_labels) in enumerate(
        zip(token_chunks, label_chunks, strict=True)
    ):
        synchronize = index == accumulation_steps - 1
        synchronization_context: Any
        if reducer is not None:
            reducer.begin_backward(sync=synchronize)
            synchronization_context = contextlib.nullcontext()
        else:
            assert isinstance(model, DDP)
            synchronization_context = contextlib.nullcontext() if synchronize else model.no_sync()
        with synchronization_context, _autocast(tokens.device, precision):
            logits = model(micro_tokens)
            loss = nn.functional.cross_entropy(logits, micro_labels)
            backward_loss = loss / accumulation_steps
        total_loss += float(loss.detach()) / accumulation_steps
        if use_scaler:
            assert scaler is not None
            scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()
        if reducer is not None:
            step_stats.append(reducer.finish_backward())

    if reducer is not None:
        reducer.assert_complete()
    base_model = model.module if isinstance(model, DDP) else model
    if before_unscale is not None:
        before_unscale(base_model)
    if use_scaler:
        assert scaler is not None
        scaler.unscale_(optimizer)
    gradient_norm: float | None = None
    if gradient_clip_norm is not None:
        gradient_norm = float(
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm).detach()
        )
    if before_optimizer_step is not None:
        before_optimizer_step(base_model)
    scale_before = scaler.get_scale() if use_scaler and scaler is not None else None
    if use_scaler:
        assert scaler is not None
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    scale_after = scaler.get_scale() if use_scaler and scaler is not None else None
    skipped = bool(
        scale_before is not None and scale_after is not None and scale_after < scale_before
    )
    timeline = [event for stats in step_stats for event in stats.timeline]
    timeline.append(
        {"kind": "optimizer_step", "timestamp_ns": time.perf_counter_ns(), "skipped": skipped}
    )
    return StepResult(
        loss=total_loss,
        gradient_norm=gradient_norm,
        optimizer_step_skipped=skipped,
        collective_count=sum(stats.collective_count for stats in step_stats),
        payload_bytes=sum(stats.payload_bytes for stats in step_stats),
        timeline=tuple(timeline),
    )
