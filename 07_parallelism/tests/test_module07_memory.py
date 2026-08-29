from __future__ import annotations

import sys
from pathlib import Path

from torch import nn

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_parallel.memory import (  # noqa: E402
    estimate_training_memory,
    minimum_parameter_count_for_oom,
)


def test_fully_sharded_persistent_state_scales_with_world_size() -> None:
    model = nn.Linear(32, 16)
    ddp = estimate_training_memory(model, activation_elements=100)
    fsdp = estimate_training_memory(
        model, world_size=4, fully_sharded=True, activation_elements=100
    )
    assert fsdp.persistent_state_bytes == ddp.persistent_state_bytes // 4
    assert fsdp.activation_lower_bound_bytes == ddp.activation_lower_bound_bytes
    assert fsdp.estimated_total_bytes > fsdp.persistent_state_bytes


def test_oom_parameter_target_is_configuration_driven() -> None:
    count = minimum_parameter_count_for_oom(
        24 * 1024**3, bytes_per_parameter=16, safety_fraction=0.9
    )
    assert count * 16 > 24 * 1024**3 * 0.9
