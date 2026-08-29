from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]


def test_small_model_fixed_batch_baseline_overfits(tmp_path: Path) -> None:
    output = tmp_path / "baseline.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_ROOT / "benchmarks" / "run_single_device_baseline.py"),
            "--config",
            str(MODULE_ROOT / "configs" / "local_baseline.toml"),
            "--output",
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["metrics"]["loss_ratio"] < 0.25
    assert payload["metrics"]["maximum_parameter_update"] > 0
