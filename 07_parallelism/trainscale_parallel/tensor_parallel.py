"""Minimal MLP and head-parallel attention using autograd-aware collectives."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import nn


def _rank_slice(size: int) -> slice:
    world_size = dist.get_world_size()
    if size % world_size:
        raise ValueError(f"dimension {size} must be divisible by world size {world_size}")
    width = size // world_size
    rank = dist.get_rank()
    return slice(rank * width, (rank + 1) * width)


class ReferenceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(inputs)))


class TensorParallelMLP(nn.Module):
    """Colwise fc1 followed by Rowwise fc2, keeping the hidden activation sharded."""

    def __init__(self, reference: ReferenceMLP) -> None:
        super().__init__()
        hidden_slice = _rank_slice(reference.fc1.out_features)
        self.hidden_slice = hidden_slice
        self.fc1_weight = nn.Parameter(reference.fc1.weight.detach()[hidden_slice].clone())
        self.fc1_bias = nn.Parameter(reference.fc1.bias.detach()[hidden_slice].clone())
        self.fc2_weight = nn.Parameter(reference.fc2.weight.detach()[:, hidden_slice].clone())
        self.output_bias = nn.Parameter(reference.fc2.bias.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        local_hidden = torch.nn.functional.gelu(
            torch.nn.functional.linear(inputs, self.fc1_weight, self.fc1_bias)
        )
        local_output = torch.nn.functional.linear(local_hidden, self.fc2_weight)
        return dist_nn.all_reduce(local_output, op=dist.ReduceOp.SUM) + self.output_bias

    def synchronize_replicated_gradients(self) -> None:
        if self.output_bias.grad is not None:
            dist.all_reduce(self.output_bias.grad, op=dist.ReduceOp.SUM)

    def local_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "fc1_weight": tuple(self.fc1_weight.shape),
            "fc2_weight": tuple(self.fc2_weight.shape),
            "output_bias": tuple(self.output_bias.shape),
        }


class ReferenceSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = inputs.shape

        def heads(projection: nn.Linear) -> torch.Tensor:
            return (
                projection(inputs)
                .view(batch, sequence, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        query, key, value = heads(self.query), heads(self.key), heads(self.value)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        context = torch.softmax(scores, dim=-1) @ value
        merged = context.transpose(1, 2).contiguous().view(batch, sequence, self.d_model)
        return self.output(merged)


class HeadParallelSelfAttention(nn.Module):
    """Shard Q/K/V heads and the output projection input dimension across ranks."""

    def __init__(self, reference: ReferenceSelfAttention) -> None:
        super().__init__()
        world_size = dist.get_world_size()
        if reference.num_heads % world_size:
            raise ValueError("num_heads must be divisible by world size")
        width_slice = _rank_slice(reference.d_model)
        self.width_slice = width_slice
        self.local_heads = reference.num_heads // world_size
        self.head_dim = reference.head_dim
        self.query_weight = nn.Parameter(reference.query.weight.detach()[width_slice].clone())
        self.query_bias = nn.Parameter(reference.query.bias.detach()[width_slice].clone())
        self.key_weight = nn.Parameter(reference.key.weight.detach()[width_slice].clone())
        self.key_bias = nn.Parameter(reference.key.bias.detach()[width_slice].clone())
        self.value_weight = nn.Parameter(reference.value.weight.detach()[width_slice].clone())
        self.value_bias = nn.Parameter(reference.value.bias.detach()[width_slice].clone())
        self.output_weight = nn.Parameter(reference.output.weight.detach()[:, width_slice].clone())
        self.output_bias = nn.Parameter(reference.output.bias.detach().clone())

    def _project(self, inputs: torch.Tensor, name: str) -> torch.Tensor:
        weight = getattr(self, f"{name}_weight")
        bias = getattr(self, f"{name}_bias")
        batch, sequence, _ = inputs.shape
        return (
            torch.nn.functional.linear(inputs, weight, bias)
            .view(batch, sequence, self.local_heads, self.head_dim)
            .transpose(1, 2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = self._project(inputs, "query")
        key = self._project(inputs, "key")
        value = self._project(inputs, "value")
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        context = torch.softmax(scores, dim=-1) @ value
        batch, _, sequence, _ = context.shape
        local_context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, sequence, self.local_heads * self.head_dim)
        )
        local_output = torch.nn.functional.linear(local_context, self.output_weight)
        return dist_nn.all_reduce(local_output, op=dist.ReduceOp.SUM) + self.output_bias

    def synchronize_replicated_gradients(self) -> None:
        if self.output_bias.grad is not None:
            dist.all_reduce(self.output_bias.grad, op=dist.ReduceOp.SUM)

    def local_shapes(self) -> dict[str, tuple[int, ...] | int]:
        return {
            "query_weight": tuple(self.query_weight.shape),
            "output_weight": tuple(self.output_weight.shape),
            "local_heads": self.local_heads,
        }
