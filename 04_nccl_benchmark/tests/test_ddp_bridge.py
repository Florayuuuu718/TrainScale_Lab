from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = MODULE_ROOT / "benchmarks" / "run_ddp_bridge.py"
SPEC = importlib.util.spec_from_file_location("module04_bridge", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_torchrun_command_is_one_process_per_gpu_and_preserves_workload() -> None:
    config = {
        "seed": 1,
        "input_dim": 1024,
        "hidden_dim": 2048,
        "num_classes": 256,
        "per_rank_batch_size": 128,
        "warmup_steps": 5,
        "profile_steps": 10,
        "bucket_cap_mb": 25.0,
        "learning_rate": 0.01,
    }
    command = MODULE.torchrun_command(
        world_size=4,
        config=config,
        rank_directory=Path("ranks"),
        trace_directory=Path("traces"),
        timeout_seconds=300,
    )
    assert "--nproc-per-node=4" in command
    assert command[command.index("--input-dim") + 1] == "1024"
    assert command[command.index("--bucket-cap-mb") + 1] == "25.0"

