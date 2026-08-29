"""Transparent lower-bound memory model for DDP and fully sharded training state."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn


@dataclass(frozen=True)
class MemoryEstimate:
    parameter_count: int
    parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    master_parameter_bytes: int
    persistent_state_bytes: int
    activation_lower_bound_bytes: int
    estimated_total_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def estimate_training_memory(
    model: nn.Module,
    *,
    world_size: int = 1,
    fully_sharded: bool = False,
    parameter_bytes: int = 4,
    gradient_bytes: int = 4,
    optimizer_state_bytes_per_parameter: int = 8,
    master_parameter_bytes: int = 0,
    activation_elements: int = 0,
    activation_bytes: int = 4,
) -> MemoryEstimate:
    """Estimate persistent state plus an explicitly supplied activation lower bound."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    byte_fields = (
        parameter_bytes,
        gradient_bytes,
        optimizer_state_bytes_per_parameter,
        master_parameter_bytes,
        activation_bytes,
    )
    if any(value < 0 for value in byte_fields) or activation_elements < 0:
        raise ValueError("memory coefficients and activation_elements must be non-negative")
    count = sum(parameter.numel() for parameter in model.parameters())
    divisor = world_size if fully_sharded else 1
    parameters = count * parameter_bytes // divisor
    gradients = count * gradient_bytes // divisor
    optimizer = count * optimizer_state_bytes_per_parameter // divisor
    master = count * master_parameter_bytes // divisor
    activation = activation_elements * activation_bytes
    persistent = parameters + gradients + optimizer + master
    return MemoryEstimate(
        parameter_count=count,
        parameter_bytes=parameters,
        gradient_bytes=gradients,
        optimizer_state_bytes=optimizer,
        master_parameter_bytes=master,
        persistent_state_bytes=persistent,
        activation_lower_bound_bytes=activation,
        estimated_total_bytes=persistent + activation,
    )


def minimum_parameter_count_for_oom(
    capacity_bytes: int,
    *,
    bytes_per_parameter: int = 16,
    safety_fraction: float = 0.9,
) -> int:
    if capacity_bytes <= 0 or bytes_per_parameter <= 0:
        raise ValueError("capacity and bytes_per_parameter must be positive")
    if not 0 < safety_fraction < 1:
        raise ValueError("safety_fraction must be between zero and one")
    return int(capacity_bytes * safety_fraction // bytes_per_parameter) + 1
