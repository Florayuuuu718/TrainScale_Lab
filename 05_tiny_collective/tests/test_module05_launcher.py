from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = MODULE_ROOT / "benchmarks" / "launcher.py"
SPEC = importlib.util.spec_from_file_location("module05_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_torchrun_command_uses_one_worker_per_rank(tmp_path: Path) -> None:
    command = MODULE.torchrun_command(4, tmp_path, ["--mode", "correctness"])
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--standalone" in command
    assert "--nproc-per-node=4" in command
    assert command[-2:] == ["--mode", "correctness"]
