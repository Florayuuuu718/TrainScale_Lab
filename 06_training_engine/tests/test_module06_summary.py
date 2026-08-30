from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = MODULE_ROOT / "benchmarks" / "summarize_module06.py"
SPEC = importlib.util.spec_from_file_location("module06_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_local_success_is_not_misreported_as_gpu_complete() -> None:
    baseline = {"status": "success", "metrics": {"loss_ratio": 0.01}}
    correctness = {
        "status": "success",
        "metrics": {
            "records": [{"ranks": [{"gradient_max_error": 1e-8, "parameter_max_error": 2e-8}]}]
        },
    }
    result = MODULE.summarize(baseline, correctness, None, None, None)
    assert result["status"] == "passed_local_gates"
    assert result["gates"]["gpu_ablation"] == "not_provided"
    assert result["pending_gpu_gates"]


def test_all_successful_gates_complete_module() -> None:
    local = {"status": "success", "metrics": {"records": []}}
    result = MODULE.summarize(
        local,
        local,
        {"status": "success"},
        {"status": "success"},
        {"status": "success"},
    )
    assert result["status"] == "complete"
    assert result["pending_gpu_gates"] == []
    assert "GPU ablation" in result["boundary"]
