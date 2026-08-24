from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def test_vector_add_relu_and_matmul_reference_shapes() -> None:
    x = torch.linspace(-2.0, 2.0, 17)
    y = torch.linspace(1.0, -1.0, 17)
    torch.testing.assert_close(torch.relu(x + y), torch.maximum(x + y, torch.tensor(0.0)))

    left = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    right = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    explicit = torch.stack(
        [
            torch.stack([(left[row] * right[:, col]).sum() for col in range(4)])
            for row in range(3)
        ]
    )
    torch.testing.assert_close(left @ right, explicit)


def test_stable_softmax_handles_large_values_and_ragged_width() -> None:
    x = torch.tensor([[10_000.0, 10_001.0, 9_999.0, -10_000.0, 0.0]])
    shifted = x - x.max(dim=-1, keepdim=True).values
    stable = shifted.exp() / shifted.exp().sum(dim=-1, keepdim=True)
    assert torch.isfinite(stable).all()
    torch.testing.assert_close(stable.sum(dim=-1), torch.ones(1))
    torch.testing.assert_close(stable, torch.softmax(x, dim=-1))


def test_layer_norm_formula_matches_pytorch() -> None:
    torch.manual_seed(7)
    x = torch.randn(4, 7)
    weight = torch.randn(7)
    bias = torch.randn(7)
    eps = 1e-5
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    expected = (x - mean) * torch.rsqrt(variance + eps) * weight + bias
    torch.testing.assert_close(
        expected, functional.layer_norm(x, (7,), weight, bias, eps), atol=1e-6, rtol=1e-6
    )


def test_explicit_attention_matches_sdpa() -> None:
    torch.manual_seed(11)
    query = torch.randn(1, 2, 5, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    explicit = torch.softmax(scores, dim=-1) @ value
    torch.testing.assert_close(
        explicit,
        functional.scaled_dot_product_attention(query, key, value),
        atol=1e-6,
        rtol=1e-6,
    )
