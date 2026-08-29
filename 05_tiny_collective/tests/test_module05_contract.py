from __future__ import annotations

import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_collective.contract import (  # noqa: E402
    load_benchmark_config,
    load_correctness_config,
)


def test_repository_configs_freeze_cpu_and_gpu_scope() -> None:
    cpu = load_correctness_config(MODULE_ROOT / "configs" / "cpu_correctness.toml")
    gpu = load_benchmark_config(MODULE_ROOT / "configs" / "gpu_comparison.toml")
    assert cpu["world_sizes"] == [2, 3, 4]
    assert 17 in cpu["element_counts"]
    assert set(cpu["algorithms"]) == {"centralized", "ring"}
    assert gpu["world_sizes"] == [2, 4]
    assert set(gpu["algorithms"]) == {"centralized", "ring", "torch"}
    assert gpu["repetitions"] == 3
    assert 10_494_976 in gpu["message_bytes"]
