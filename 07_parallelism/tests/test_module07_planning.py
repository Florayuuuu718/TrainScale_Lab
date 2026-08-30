from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_plan_is_targeted_and_includes_reference() -> None:
    runner = _load(MODULE_ROOT / "benchmarks" / "run_gpu_parallelism.py", "module07_gpu_runner")
    config = runner.load_gpu_config(MODULE_ROOT / "configs" / "gpu_parallelism.toml")
    cases = runner.planned_cases(config)
    assert len(cases) == 11
    assert {case["strategy"] for case in cases} == {
        "ddp",
        "fsdp_root",
        "fsdp_layer",
        "tp",
        "tp_reference",
    }
    assert len(runner.planned_preflights(config)) == 4


def test_acceptance_distinguishes_local_from_gpu_completion() -> None:
    summary = _load(MODULE_ROOT / "benchmarks" / "summarize_module07.py", "module07_summary")
    success = {"status": "success", "metrics": {"records": [], "ranks": []}}
    payload = summary.summarize(success, success, success, success, None, None)
    assert payload["status"] == "passed_local_gates"
    assert payload["gates"]["gpu_parallelism"] == "not_provided"


def test_cuda_preflight_can_replace_unavailable_cpu_fsdp_gate() -> None:
    summary = _load(MODULE_ROOT / "benchmarks" / "summarize_module07.py", "module07_gpu_summary")
    success = {"status": "success", "metrics": {"records": [], "ranks": []}}
    unavailable = {"status": "unavailable", "metrics": {"ranks": []}}
    gpu = {
        "status": "success",
        "correctness": {
            "preflights": [
                {"mode": "fsdp2_probe", "status": "success"},
                {"mode": "native_tp_probe", "status": "success"},
            ]
        },
    }
    payload = summary.summarize(success, success, unavailable, success, gpu, success)
    assert payload["status"] == "complete"
    assert payload["gates"]["fsdp2_cpu_gloo"] == "unavailable"
    assert payload["gates"]["fsdp2_cuda_nccl_preflight"] == "success"
