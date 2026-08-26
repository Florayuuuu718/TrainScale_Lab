from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform == "win32", reason="Linux CPU CI runs the torchrun integration")
def test_two_rank_gloo_semantics_through_public_runner(tmp_path: Path) -> None:
    output = tmp_path / "semantics.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_ROOT / "benchmarks" / "run_correctness.py"),
            "--experiment",
            "semantics",
            "--world-size",
            "2",
            "--output",
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["all_checks_passed"] is True
    assert {rank["rank"] for rank in payload["result"]["ranks"]} == {0, 1}

