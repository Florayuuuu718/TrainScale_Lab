from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

pytest.importorskip("triton", reason="Triton GPU tests require the Linux CUDA environment")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainscale_kernels import (  # noqa: E402
    attention,
    layer_norm,
    layer_norm_backward,
    matmul,
    matmul_backward,
    matmul_configured,
    relu_add,
    relu_add_backward,
    softmax,
    softmax_baseline,
    vector_add,
)

_IS_SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
_SM120_LAUNCH_ENABLED = os.environ.get("TRAINSCALE_RUN_SM120_TRITON") == "1"
_REQUIRES_SM120_OPT_IN = pytest.mark.skipif(
    _IS_SM120 and not _SM120_LAUNCH_ENABLED,
    reason=(
        "SM 12.0 Triton launches require the crash-isolated environment probe first; "
        "set TRAINSCALE_RUN_SM120_TRITON=1 only after it passes"
    ),
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@_REQUIRES_SM120_OPT_IN
@pytest.mark.parametrize("size", [17, 257, 4097])
def test_vector_add_handles_ragged_sizes(size: int) -> None:
    x = torch.randn(size, device="cuda")
    y = torch.randn_like(x)
    torch.testing.assert_close(vector_add(x, y), x + y)
    output = torch.empty_like(x)
    assert vector_add(x, y, out=output).data_ptr() == output.data_ptr()
    torch.testing.assert_close(output, x + y)


@_REQUIRES_SM120_OPT_IN
def test_relu_add_forward_and_backward() -> None:
    x = torch.linspace(-2, 2, 509, device="cuda")
    bias = torch.full_like(x, 0.25)
    grad = torch.randn_like(x)
    actual = relu_add(x, bias)
    expected = torch.relu(x + bias)
    torch.testing.assert_close(actual, expected)
    dx, dbias = relu_add_backward(grad, x, bias)
    expected_grad = torch.where(x + bias > 0, grad, 0.0)
    torch.testing.assert_close(dx, expected_grad)
    torch.testing.assert_close(dbias, expected_grad)


@_REQUIRES_SM120_OPT_IN
@pytest.mark.parametrize("shape", [(1, 17), (32, 127), (8, 509)])
def test_softmax_is_stable_for_ragged_rows(shape: tuple[int, int]) -> None:
    x = torch.randn(shape, device="cuda") * 20
    torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1), atol=2e-6, rtol=2e-5)
    output = torch.empty_like(x)
    assert softmax(x, out=output).data_ptr() == output.data_ptr()
    torch.testing.assert_close(output, torch.softmax(x, dim=-1), atol=2e-6, rtol=2e-5)
    baseline_output = torch.empty_like(x)
    assert softmax_baseline(x, out=baseline_output).data_ptr() == baseline_output.data_ptr()
    torch.testing.assert_close(
        baseline_output, torch.softmax(x, dim=-1), atol=2e-6, rtol=2e-5
    )


@_REQUIRES_SM120_OPT_IN
def test_layer_norm_forward_and_backward() -> None:
    rows, hidden = 8, 127
    x = torch.randn((rows, hidden), device="cuda")
    weight = torch.randn(hidden, device="cuda")
    bias = torch.randn(hidden, device="cuda")
    grad = torch.randn_like(x)

    expected_x = x.detach().clone().requires_grad_(True)
    expected_weight = weight.detach().clone().requires_grad_(True)
    expected_bias = bias.detach().clone().requires_grad_(True)
    expected = functional.layer_norm(
        expected_x, (hidden,), expected_weight, expected_bias, eps=1e-5
    )
    expected.backward(grad)

    actual, mean, rstd = layer_norm(x, weight, bias)
    dx, dweight, dbias = layer_norm_backward(grad, x, weight, mean, rstd)
    torch.testing.assert_close(actual, expected.detach(), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(dx, expected_x.grad, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(dweight, expected_weight.grad, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(dbias, expected_bias.grad, atol=5e-5, rtol=5e-5)


@_REQUIRES_SM120_OPT_IN
@pytest.mark.parametrize("shape", [(17, 31, 23), (64, 64, 64)])
def test_matmul_forward_and_backward(shape: tuple[int, int, int]) -> None:
    m_size, n_size, k_size = shape
    a = torch.randn((m_size, k_size), device="cuda", dtype=torch.float16)
    b = torch.randn((k_size, n_size), device="cuda", dtype=torch.float16)
    grad = torch.randn((m_size, n_size), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(matmul(a, b), a @ b, atol=2e-2, rtol=2e-2)
    da, db = matmul_backward(grad, a, b)
    torch.testing.assert_close(da, grad @ b.T, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(db, a.T @ grad, atol=2e-2, rtol=2e-2)
    candidates = (
        {"block_m": 32, "block_n": 32, "block_k": 32, "group_m": 8, "num_warps": 4},
        {"block_m": 64, "block_n": 32, "block_k": 32, "group_m": 8, "num_warps": 4},
        {"block_m": 32, "block_n": 64, "block_k": 32, "group_m": 8, "num_warps": 4},
        {"block_m": 64, "block_n": 64, "block_k": 32, "group_m": 8, "num_warps": 8},
    )
    for candidate in candidates:
        torch.testing.assert_close(
            matmul_configured(a, b, **candidate), a @ b, atol=2e-2, rtol=2e-2
        )


@_REQUIRES_SM120_OPT_IN
@pytest.mark.parametrize(
    ("causal", "head_dim"), [(False, 16), (True, 32), (False, 64), (True, 128)]
)
def test_attention_forward(causal: bool, head_dim: int) -> None:
    heads, sequence = 2, 33
    query = torch.randn((heads, sequence, head_dim), device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    expected = functional.scaled_dot_product_attention(
        query[None], key[None], value[None], is_causal=causal
    )[0]
    torch.testing.assert_close(
        attention(query, key, value, causal=causal), expected, atol=3e-2, rtol=3e-2
    )


def test_unsupported_inputs_are_rejected() -> None:
    cpu = torch.ones(4)
    with pytest.raises(ValueError, match="CUDA"):
        vector_add(cpu, cpu)

    cuda = torch.ones((2, 300, 32), device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="sequence <= 256"):
        attention(cuda, cuda, cuda)

    bad_head = torch.ones((2, 33, 24), device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="head_dim"):
        attention(bad_head, bad_head, bad_head)

    too_wide = torch.ones((1, 65_537), device="cuda")
    with pytest.raises(ValueError, match="65,536"):
        softmax(too_wide)
