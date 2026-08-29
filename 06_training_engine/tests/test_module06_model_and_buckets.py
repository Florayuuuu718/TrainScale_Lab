from __future__ import annotations

import sys
from pathlib import Path

import torch

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_engine.bucket import bucket_plan_digest, build_bucket_plan  # noqa: E402
from trainscale_engine.model import TinyTransformer, model_preset  # noqa: E402


def test_tiny_transformer_presets_are_scalable_and_shape_correct() -> None:
    small = TinyTransformer(model_preset("small"))
    medium = TinyTransformer(model_preset("medium"))
    tokens = torch.zeros((2, 8), dtype=torch.long)
    assert small(tokens).shape == (2, 4)
    assert sum(parameter.numel() for parameter in medium.parameters()) > sum(
        parameter.numel() for parameter in small.parameters()
    )


def test_bucket_plan_owns_every_parameter_once_with_non_overlapping_offsets() -> None:
    model = TinyTransformer(model_preset("small"))
    plan = build_bucket_plan(model, bucket_cap_bytes=1024)
    expected = set(dict(model.named_parameters()))
    actual = [entry.name for bucket in plan for entry in bucket.entries]
    assert set(actual) == expected
    assert len(actual) == len(set(actual))
    for index, bucket in enumerate(plan):
        assert bucket.index == index
        assert sum(entry.numel for entry in bucket.entries) == bucket.numel
        assert [entry.offset for entry in bucket.entries] == list(
            torch.tensor([0, *[entry.numel for entry in bucket.entries[:-1]]]).cumsum(0).tolist()
        )
    assert bucket_plan_digest(plan) == bucket_plan_digest(
        build_bucket_plan(model, bucket_cap_bytes=1024)
    )
