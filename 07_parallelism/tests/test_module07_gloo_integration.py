from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform == "win32", reason="Linux CPU CI runs Gloo integration")
def test_tp_correctness_runner_executes_mlp_and_attention_matrix(tmp_path: Path) -> None:
    output = tmp_path / "tp.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_ROOT / "benchmarks" / "run_tp_correctness.py"),
            "--config",
            str(MODULE_ROOT / "configs" / "local_correctness.toml"),
            "--raw-directory",
            str(tmp_path / "raw"),
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
    assert len(payload["metrics"]["records"]) == 4
    assert (
        max(
            rank["maximum_error"]
            for record in payload["metrics"]["records"]
            for rank in record["ranks"]
        )
        <= payload["config"]["atol"]
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Linux CPU CI runs Gloo integration")
@pytest.mark.parametrize(
    ("script", "artifact_type", "backend_capability_may_be_unavailable"),
    [
        ("run_fsdp2_capability.py", "module07.fsdp2_cpu_capability", True),
        ("run_native_tp_capability.py", "module07.native_tp_cpu_capability", False),
    ],
)
def test_real_pytorch_sharding_capabilities(
    tmp_path: Path,
    script: str,
    artifact_type: str,
    backend_capability_may_be_unavailable: bool,
) -> None:
    output = tmp_path / f"{script}.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_ROOT / "benchmarks" / script),
            "--raw-directory",
            str(tmp_path / f"raw-{script}"),
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
    assert payload["artifact_type"] == artifact_type
    if backend_capability_may_be_unavailable and payload["status"] == "unavailable":
        # FSDP2 reduction semantics have differed across PyTorch CPU/Gloo
        # versions. This is a capability result, not a CUDA/NCCL failure.
        assert payload["correctness"]["status"] == "not_run"
        assert payload["metrics"]["ranks"]
    else:
        assert payload["status"] == "success"
        assert all(rank["correctness_passed"] for rank in payload["metrics"]["ranks"])
