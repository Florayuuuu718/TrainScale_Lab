from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform == "win32", reason="Linux CPU CI runs Gloo integration")
def test_cpu_correctness_runner_executes_all_cases(tmp_path: Path) -> None:
    output = tmp_path / "correctness.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_ROOT / "benchmarks" / "run_correctness.py"),
            "--config",
            str(MODULE_ROOT / "configs" / "cpu_correctness.toml"),
            "--output",
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert len(payload["metrics"]["records"]) == 24
