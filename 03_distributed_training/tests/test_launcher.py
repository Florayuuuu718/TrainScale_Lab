from __future__ import annotations

import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from benchmarks.launcher import torchrun_command  # noqa: E402


def test_torchrun_command_uses_standalone_and_requested_world_size(tmp_path: Path) -> None:
    command = torchrun_command(4, tmp_path, ["--mode", "semantics"])
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--standalone" in command
    assert "--nproc-per-node=4" in command
    assert command[-2:] == ["--mode", "semantics"]

