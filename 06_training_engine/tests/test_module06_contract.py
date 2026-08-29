from __future__ import annotations

import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_engine.contract import (  # noqa: E402
    load_baseline_config,
    load_benchmark_config,
    load_correctness_config,
)


def test_repository_configs_freeze_local_and_gpu_scope() -> None:
    local = load_correctness_config(MODULE_ROOT / "configs" / "local_correctness.toml")
    gpu = load_benchmark_config(MODULE_ROOT / "configs" / "gpu_ablation.toml")
    baseline = load_baseline_config(MODULE_ROOT / "configs" / "local_baseline.toml")
    assert set(local["strategies"]) == {
        "bulk",
        "per_parameter",
        "bucket_sync",
        "bucket_async",
        "ddp",
    }
    assert local["accumulation_steps"] == [1, 2]
    assert local["include_unused"] is True
    assert gpu["world_sizes"] == [2, 4]
    assert len(gpu["bucket_cap_mb"]) == 3
    assert set(gpu["precisions"]) == {"fp32", "amp"}
    assert baseline["device"] == "cpu"
