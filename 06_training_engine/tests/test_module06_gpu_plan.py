from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(MODULE_ROOT.parent / "benchmarks"))
SCRIPT = MODULE_ROOT / "benchmarks" / "run_gpu_ablation.py"
SPEC = importlib.util.spec_from_file_location("module06_gpu_plan", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
OVERFLOW_SCRIPT = MODULE_ROOT / "benchmarks" / "run_amp_overflow_probe.py"
OVERFLOW_SPEC = importlib.util.spec_from_file_location(
    "module06_amp_overflow_plan", OVERFLOW_SCRIPT
)
assert OVERFLOW_SPEC is not None and OVERFLOW_SPEC.loader is not None
OVERFLOW_MODULE = importlib.util.module_from_spec(OVERFLOW_SPEC)
OVERFLOW_SPEC.loader.exec_module(OVERFLOW_MODULE)

from trainscale_engine.contract import load_benchmark_config  # noqa: E402


def test_targeted_gpu_plan_covers_each_axis_without_cartesian_explosion() -> None:
    config = load_benchmark_config(MODULE_ROOT / "configs" / "gpu_ablation.toml")
    cases = MODULE.planned_cases(config)
    assert len(cases) == 20
    assert len(cases) < 50
    assert {case["strategy"] for case in cases} == set(config["strategies"])
    assert {case["world_size"] for case in cases} == {2, 4}
    assert {case["model_preset"] for case in cases} == {"small", "medium"}
    assert {case["precision"] for case in cases} == {"fp32", "amp"}
    assert {case["accumulation_steps"] for case in cases} == {1, 4}
    assert {case["bucket_cap_mb"] for case in cases} == set(config["bucket_cap_mb"])
    assert any(case["bucket_cap_bytes"] == 10_494_976 for case in cases)


def test_amp_overflow_plan_covers_manual_and_ddp_paths_on_two_world_sizes() -> None:
    config = load_benchmark_config(MODULE_ROOT / "configs" / "gpu_ablation.toml")
    cases = OVERFLOW_MODULE.planned_cases(config)
    assert len(cases) == 4
    assert {case["world_size"] for case in cases} == {2, 4}
    assert {case["strategy"] for case in cases} == {"bucket_async", "ddp"}
