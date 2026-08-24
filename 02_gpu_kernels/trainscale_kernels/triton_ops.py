"""Readable Triton implementations used by the module 02 experiments."""

from __future__ import annotations

import math
from typing import Final

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]

_FLOAT_DTYPES: Final = (torch.float16, torch.bfloat16, torch.float32)


def _require_cuda_contiguous(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if not tensor.is_cuda:
            raise ValueError("Triton kernels require CUDA tensors")
        if not tensor.is_contiguous():
            raise ValueError("This teaching implementation requires contiguous tensors")
        if tensor.dtype not in _FLOAT_DTYPES:
            raise TypeError(f"Unsupported dtype: {tensor.dtype}")


def _require_same_shape(*tensors: torch.Tensor) -> None:
    first = tensors[0]
    if any(tensor.shape != first.shape for tensor in tensors[1:]):
        raise ValueError("All tensors must have the same shape")
    if any(tensor.dtype != first.dtype for tensor in tensors[1:]):
        raise TypeError("All tensors must have the same dtype")
    if any(tensor.device != first.device for tensor in tensors[1:]):
        raise ValueError("All tensors must be on the same device")


@triton.jit
def _vector_add_kernel(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    output_ptr: tl.tensor,
    size: tl.int32,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def vector_add(
    x: torch.Tensor,
    y: torch.Tensor,
    block_size: int = 128,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _require_cuda_contiguous(x, y)
    _require_same_shape(x, y)
    if x.numel() == 0:
        return torch.empty_like(x)
    if block_size not in {128, 256, 512, 1024}:
        raise ValueError("block_size must be one of 128, 256, 512, 1024")
    if out is not None:
        _require_cuda_contiguous(out)
        _require_same_shape(x, out)
    output = torch.empty_like(x) if out is None else out
    grid = (triton.cdiv(x.numel(), block_size),)
    _vector_add_kernel[grid](
        x, y, output, x.numel(), BLOCK_SIZE=block_size, num_warps=1
    )
    return output


@triton.jit
def _relu_add_kernel(
    x_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    output_ptr: tl.tensor,
    size: tl.int32,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    value = tl.load(x_ptr + offsets, mask=mask) + tl.load(bias_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.maximum(value, 0.0), mask=mask)


@triton.jit
def _relu_add_backward_kernel(
    grad_ptr: tl.tensor,
    x_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    dx_ptr: tl.tensor,
    dbias_ptr: tl.tensor,
    size: tl.int32,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    grad = tl.load(grad_ptr + offsets, mask=mask)
    value = tl.load(x_ptr + offsets, mask=mask) + tl.load(bias_ptr + offsets, mask=mask)
    result = tl.where(value > 0.0, grad, 0.0)
    tl.store(dx_ptr + offsets, result, mask=mask)
    tl.store(dbias_ptr + offsets, result, mask=mask)


def relu_add(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    _require_cuda_contiguous(x, bias)
    _require_same_shape(x, bias)
    output = torch.empty_like(x)
    if x.numel():
        _relu_add_kernel[(triton.cdiv(x.numel(), 128),)](
            x, bias, output, x.numel(), BLOCK_SIZE=128, num_warps=1
        )
    return output


def relu_add_backward(
    grad_output: torch.Tensor, x: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_cuda_contiguous(grad_output, x, bias)
    _require_same_shape(grad_output, x, bias)
    dx = torch.empty_like(x)
    dbias = torch.empty_like(bias)
    if x.numel():
        _relu_add_backward_kernel[(triton.cdiv(x.numel(), 128),)](
            grad_output,
            x,
            bias,
            dx,
            dbias,
            x.numel(),
            BLOCK_SIZE=128,
            num_warps=1,
        )
    return dx, dbias


@triton.jit
def _softmax_kernel(
    input_ptr: tl.tensor,
    output_ptr: tl.tensor,
    input_row_stride: tl.int64,
    output_row_stride: tl.int64,
    cols: tl.int32,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < cols
    values = tl.load(input_ptr + row * input_row_stride + offsets, mask=mask, other=-float("inf"))
    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    denominator = tl.sum(numerator, axis=0)
    tl.store(
        output_ptr + row * output_row_stride + offsets,
        numerator / denominator,
        mask=mask,
    )


def _softmax_launch(
    x: torch.Tensor, *, out: torch.Tensor | None, num_warps: int
) -> torch.Tensor:
    _require_cuda_contiguous(x)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("softmax expects a non-empty 2D tensor")
    block_size = triton.next_power_of_2(x.shape[1])
    if block_size > 65536:
        raise ValueError("This teaching softmax supports at most 65,536 columns")
    if out is not None:
        _require_cuda_contiguous(out)
        _require_same_shape(x, out)
    output = torch.empty_like(x) if out is None else out
    _softmax_kernel[(x.shape[0],)](
        x,
        output,
        x.stride(0),
        output.stride(0),
        x.shape[1],
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def softmax_baseline(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Use the same stable math with a deliberately fixed single-warp launch."""
    return _softmax_launch(x, out=out, num_warps=1)


def softmax(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Use a simple row-width-aware warp policy for the teaching optimized path."""
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("softmax expects a non-empty 2D tensor")
    block_size = triton.next_power_of_2(x.shape[1])
    return _softmax_launch(x, out=out, num_warps=4 if block_size < 2048 else 8)


@triton.jit
def _layer_norm_forward_kernel(
    x_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    output_ptr: tl.tensor,
    mean_ptr: tl.tensor,
    rstd_ptr: tl.tensor,
    hidden: tl.int32,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden
    x = tl.load(x_ptr + row * hidden + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / hidden
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / hidden
    rstd = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    output = centered * rstd * weight + bias
    tl.store(output_ptr + row * hidden + offsets, output, mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _layer_norm_backward_kernel(
    grad_ptr: tl.tensor,
    x_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    mean_ptr: tl.tensor,
    rstd_ptr: tl.tensor,
    dx_ptr: tl.tensor,
    dweight_ptr: tl.tensor,
    dbias_ptr: tl.tensor,
    hidden: tl.int32,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden
    x = tl.load(x_ptr + row * hidden + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + row * hidden + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    xhat = tl.where(mask, (x - mean) * rstd, 0.0)
    weighted_grad = grad * weight
    mean_weighted_grad = tl.sum(weighted_grad, axis=0) / hidden
    mean_xhat_weighted_grad = tl.sum(xhat * weighted_grad, axis=0) / hidden
    dx = (weighted_grad - mean_weighted_grad - xhat * mean_xhat_weighted_grad) * rstd
    tl.store(dx_ptr + row * hidden + offsets, dx, mask=mask)
    tl.atomic_add(dweight_ptr + offsets, grad * xhat, mask=mask)
    tl.atomic_add(dbias_ptr + offsets, grad, mask=mask)


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_cuda_contiguous(x, weight, bias)
    if x.ndim != 2 or weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("layer_norm expects x[rows, hidden], weight[hidden], bias[hidden]")
    if x.shape[1] != weight.numel() or weight.shape != bias.shape:
        raise ValueError("weight and bias must match the hidden dimension")
    if x.dtype != weight.dtype or x.dtype != bias.dtype:
        raise TypeError("x, weight, and bias must have the same dtype")
    block_size = triton.next_power_of_2(x.shape[1])
    if block_size > 65536:
        raise ValueError("This teaching LayerNorm supports hidden <= 65,536")
    output = torch.empty_like(x)
    mean = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    _layer_norm_forward_kernel[(x.shape[0],)](
        x,
        weight,
        bias,
        output,
        mean,
        rstd,
        x.shape[1],
        eps,
        BLOCK_SIZE=block_size,
        num_warps=4 if block_size < 2048 else 8,
    )
    return output, mean, rstd


def layer_norm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_cuda_contiguous(grad_output, x, weight, mean, rstd)
    if grad_output.shape != x.shape or x.ndim != 2:
        raise ValueError("grad_output and x must have the same 2D shape")
    block_size = triton.next_power_of_2(x.shape[1])
    dx = torch.empty_like(x)
    dweight = torch.zeros(x.shape[1], device=x.device, dtype=torch.float32)
    dbias = torch.zeros_like(dweight)
    _layer_norm_backward_kernel[(x.shape[0],)](
        grad_output,
        x,
        weight,
        mean,
        rstd,
        dx,
        dweight,
        dbias,
        x.shape[1],
        BLOCK_SIZE=block_size,
        num_warps=4 if block_size < 2048 else 8,
    )
    return dx, dweight, dbias


@triton.jit
def _matmul_kernel(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
    m_size: tl.int32,
    n_size: tl.int32,
    k_size: tl.int32,
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
) -> None:
    program_id = tl.program_id(0)
    programs_m = tl.cdiv(m_size, BLOCK_M)
    programs_n = tl.cdiv(n_size, BLOCK_N)
    programs_in_group = GROUP_M * programs_n
    group_id = program_id // programs_in_group
    first_program_m = group_id * GROUP_M
    group_size_m = tl.minimum(programs_m - first_program_m, GROUP_M)
    program_m = first_program_m + ((program_id % programs_in_group) % group_size_m)
    program_n = (program_id % programs_in_group) // group_size_m

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    a_offsets = offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_offsets = offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(k_size, BLOCK_K)):
        k_offsets = k_start * BLOCK_K + offsets_k
        a = tl.load(
            a_ptr + a_offsets + k_start * BLOCK_K * stride_ak,
            mask=(offsets_m[:, None] < m_size) & (k_offsets[None, :] < k_size),
            other=0.0,
        )
        b = tl.load(
            b_ptr + b_offsets + k_start * BLOCK_K * stride_bk,
            mask=(k_offsets[:, None] < k_size) & (offsets_n[None, :] < n_size),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
    c_offsets = offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(
        c_ptr + c_offsets,
        accumulator,
        mask=(offsets_m[:, None] < m_size) & (offsets_n[None, :] < n_size),
    )


def matmul_configured(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    group_m: int,
    num_warps: int,
) -> torch.Tensor:
    _require_cuda_contiguous(a, b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("matmul expects a[M,K] and b[K,N]")
    if a.dtype != b.dtype:
        raise TypeError("a and b must have the same dtype")
    m_size, k_size = a.shape
    _, n_size = b.shape
    output = torch.empty((m_size, n_size), device=a.device, dtype=a.dtype)
    if block_m not in {16, 32, 64, 128} or block_n not in {16, 32, 64, 128}:
        raise ValueError("block_m and block_n must be one of 16, 32, 64, 128")
    if block_k not in {16, 32, 64}:
        raise ValueError("block_k must be one of 16, 32, 64")
    if group_m <= 0 or num_warps not in {1, 2, 4, 8}:
        raise ValueError("group_m must be positive and num_warps must be 1, 2, 4, or 8")
    grid = (triton.cdiv(m_size, block_m) * triton.cdiv(n_size, block_n),)
    _matmul_kernel[grid](
        a,
        b,
        output,
        m_size,
        n_size,
        k_size,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
    )
    return output


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return matmul_configured(
        a,
        b,
        block_m=32,
        block_n=32,
        block_k=32,
        group_m=8,
        num_warps=4,
    )


def matmul_backward(
    grad_output: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_cuda_contiguous(grad_output, a, b)
    da = matmul(grad_output, b.transpose(0, 1).contiguous())
    db = matmul(a.transpose(0, 1).contiguous(), grad_output)
    return da, db


@triton.jit
def _attention_forward_kernel(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    v_ptr: tl.tensor,
    output_ptr: tl.tensor,
    sequence: tl.int32,
    head_dim: tl.int32,
    scale: tl.float32,
    stride_qh: tl.int64,
    stride_qq: tl.int64,
    stride_qd: tl.int64,
    stride_kh: tl.int64,
    stride_ks: tl.int64,
    stride_kd: tl.int64,
    stride_vh: tl.int64,
    stride_vs: tl.int64,
    stride_vd: tl.int64,
    stride_oh: tl.int64,
    stride_oq: tl.int64,
    stride_od: tl.int64,
    CAUSAL: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    query_index = tl.program_id(0)
    head_index = tl.program_id(1)
    sequence_offsets = tl.arange(0, BLOCK_S)
    dim_offsets = tl.arange(0, BLOCK_D)
    query = tl.load(
        q_ptr + head_index * stride_qh + query_index * stride_qq + dim_offsets * stride_qd,
        mask=dim_offsets < head_dim,
        other=0.0,
    ).to(tl.float32)
    keys = tl.load(
        k_ptr
        + head_index * stride_kh
        + sequence_offsets[:, None] * stride_ks
        + dim_offsets[None, :] * stride_kd,
        mask=(sequence_offsets[:, None] < sequence) & (dim_offsets[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(keys * query[None, :], axis=1) * scale
    score_mask = sequence_offsets < sequence
    if CAUSAL:
        score_mask = score_mask & (sequence_offsets <= query_index)
    scores = tl.where(score_mask, scores, -float("inf"))
    probabilities = tl.exp(scores - tl.max(scores, axis=0))
    probabilities = probabilities / tl.sum(probabilities, axis=0)
    values = tl.load(
        v_ptr
        + head_index * stride_vh
        + sequence_offsets[:, None] * stride_vs
        + dim_offsets[None, :] * stride_vd,
        mask=(sequence_offsets[:, None] < sequence) & (dim_offsets[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)
    output = tl.sum(probabilities[:, None] * values, axis=0)
    tl.store(
        output_ptr
        + head_index * stride_oh
        + query_index * stride_oq
        + dim_offsets * stride_od,
        output,
        mask=dim_offsets < head_dim,
    )


def attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, causal: bool = False
) -> torch.Tensor:
    _require_cuda_contiguous(query, key, value)
    _require_same_shape(query, key, value)
    if query.ndim != 3:
        raise ValueError("attention expects [heads, sequence, head_dim]")
    heads, sequence, head_dim = query.shape
    if sequence > 256:
        raise ValueError("This teaching attention supports sequence <= 256")
    if head_dim not in {16, 32, 64, 128}:
        raise ValueError("head_dim must be one of 16, 32, 64, 128")
    output = torch.empty_like(query)
    _attention_forward_kernel[(sequence, heads)](
        query,
        key,
        value,
        output,
        sequence,
        head_dim,
        1.0 / math.sqrt(head_dim),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        CAUSAL=causal,
        BLOCK_S=triton.next_power_of_2(sequence),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=8,
    )
    return output
