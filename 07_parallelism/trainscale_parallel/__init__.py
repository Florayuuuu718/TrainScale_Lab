"""Memory, sharding, and tensor-parallel teaching primitives for Module 07."""

from .memory import estimate_training_memory
from .sharding import balanced_shard_ranges, reconstruct_tensor, shard_tensor
from .tensor_parallel import (
    HeadParallelSelfAttention,
    ReferenceMLP,
    ReferenceSelfAttention,
    TensorParallelMLP,
)

__all__ = [
    "HeadParallelSelfAttention",
    "ReferenceMLP",
    "ReferenceSelfAttention",
    "TensorParallelMLP",
    "balanced_shard_ranges",
    "estimate_training_memory",
    "reconstruct_tensor",
    "shard_tensor",
]
