from __future__ import annotations

import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_parallel.contract import load_gpu_config, load_local_config  # noqa: E402


def test_repository_configs_freeze_local_and_gpu_scope() -> None:
    local = load_local_config(MODULE_ROOT / "configs" / "local_correctness.toml")
    gpu = load_gpu_config(MODULE_ROOT / "configs" / "gpu_parallelism.toml")
    assert local["world_sizes"] == [2, 4]
    assert set(local["cases"]) == {"tp_mlp", "tp_attention"}
    assert set(gpu["strategies"]) == {"ddp", "fsdp2", "tp"}
    assert gpu["enable_2d"] is False
