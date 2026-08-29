from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform == "win32", reason="Linux CPU CI runs Gloo integration")
def test_local_correctness_runner_executes_complete_matrix(tmp_path: Path) -> None:
    output = tmp_path / "correctness.json"
    raw = tmp_path / "raw"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_ROOT / "benchmarks" / "run_reducer_correctness.py"),
            "--config",
            str(MODULE_ROOT / "configs" / "local_correctness.toml"),
            "--raw-directory",
            str(raw),
            "--output",
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert len(payload["metrics"]["records"]) == 10
    assert payload["raw_artifacts"]
    records = payload["metrics"]["records"]
    assert all(record["ranks"][0]["none_gradient_names"] == ["unused_probe"] for record in records)
    by_strategy = {}
    for record in records:
        rank = record["ranks"][0]
        by_strategy.setdefault(rank["strategy"], []).append(rank["collective_count"])
    assert all(len(set(counts)) == 1 for counts in by_strategy.values())
    assert all(
        record["ranks"][0]["overlap_candidate"]
        for record in records
        if record["ranks"][0]["strategy"] == "bucket_async"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Linux CPU CI runs Gloo integration")
def test_bucket_plan_mismatch_fails_fast() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(MODULE_ROOT)
    worker = MODULE_ROOT / "tests" / "helpers" / "bucket_mismatch_worker.py"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            str(worker),
        ],
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "bucket plan differs across ranks" in completed.stdout
