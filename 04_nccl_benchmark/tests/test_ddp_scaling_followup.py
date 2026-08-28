from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
MODULE03_ROOT = REPOSITORY_ROOT / "03_distributed_training"
sys.path.insert(0, str(MODULE03_ROOT))

from trainscale_distributed.contract import load_benchmark_config  # noqa: E402


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGGREGATOR = load_script(
    "module04_ddp_scaling_aggregate",
    MODULE_ROOT / "benchmarks" / "aggregate_ddp_scaling.py",
)
CAMPAIGN = load_script(
    "module04_ddp_scaling_campaign",
    MODULE_ROOT / "benchmarks" / "run_ddp_scaling_campaign.py",
)


def source_payload(scale: float = 1.0) -> dict[str, Any]:
    config = {
        "device": "cuda",
        "backend": "nccl",
        "world_sizes": [1, 2, 4],
        "modes": ["strong", "weak"],
        "seed": 20260824,
        "input_dim": 1024,
        "hidden_dim": 2048,
        "num_classes": 256,
        "global_batch_size": 256,
        "per_rank_batch_size": 128,
        "warmup_steps": 200,
        "measured_steps": 5000,
        "learning_rate": 0.01,
    }
    records = []
    bases = {
        ("strong", 1): 200_000.0,
        ("strong", 2): 150_000.0,
        ("strong", 4): 110_000.0,
        ("weak", 1): 100_000.0,
        ("weak", 2): 150_000.0,
        ("weak", 4): 250_000.0,
    }
    for (mode, world_size), throughput in bases.items():
        local_batch = 256 // world_size if mode == "strong" else 128
        global_batch = 256 if mode == "strong" else 128 * world_size
        records.append(
            {
                "status": "success",
                "worker_mode": "benchmark",
                "mode": mode,
                "world_size": world_size,
                "backend": "nccl",
                "device": "cuda",
                "local_batch_size": local_batch,
                "global_batch_size": global_batch,
                "warmup_steps": 200,
                "measured_steps": 5000,
                "global_samples_per_second": throughput * scale,
                "max_rank_elapsed_seconds": 5.0 / scale,
                "peak_memory_allocated_bytes": 123_456,
            }
        )
    return {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "environment": {"torch": "test", "cuda_device_count": 4},
        "config": config,
        "records": records,
        "all_executable_cases_passed": True,
    }


def test_long_config_keeps_workload_and_extends_measurement_window() -> None:
    config = load_benchmark_config(MODULE_ROOT / "configs" / "ddp_scaling_long.toml")
    assert config["world_sizes"] == [1, 2, 4]
    assert config["modes"] == ["strong", "weak"]
    assert config["warmup_steps"] == 200
    assert config["measured_steps"] == 5000
    assert config["input_dim"] == 1024
    assert config["hidden_dim"] == 2048
    assert config["num_classes"] == 256


def test_five_stable_runs_pass_quality_gate_and_recompute_speedup() -> None:
    payloads = [source_payload(scale) for scale in (0.99, 1.0, 1.01, 0.995, 1.005)]
    result = AGGREGATOR.aggregate_sources(payloads, stability_threshold=0.05)
    assert result["measurement_quality"]["status"] == "passed"
    strong_four = next(
        record
        for record in result["records"]
        if record["mode"] == "strong" and record["world_size"] == 4
    )
    assert strong_four["repeat_count"] == 5
    assert strong_four["throughput_median_samples_per_second"] == pytest.approx(110_000.0)
    assert strong_four["speedup_over_world_one_median"] == pytest.approx(0.55)
    assert strong_four["scaling_efficiency"] == pytest.approx(0.1375)


def test_unstable_case_is_recorded_without_fabricating_failure() -> None:
    payloads = [source_payload() for _ in range(5)]
    unstable = next(
        record
        for record in payloads[-1]["records"]
        if record["mode"] == "weak" and record["world_size"] == 1
    )
    unstable["global_samples_per_second"] *= 0.5
    result = AGGREGATOR.aggregate_sources(payloads, stability_threshold=0.05)
    assert result["status"] == "success"
    assert result["correctness"]["status"] == "passed"
    assert result["measurement_quality"] == {
        "status": "failed",
        "unstable_cases": [{"mode": "weak", "world_size": 1}],
        "severely_unstable_cases": [{"mode": "weak", "world_size": 1}],
        "unavailable_cases": [],
    }


def test_moderate_variation_is_a_warning() -> None:
    payloads = [source_payload() for _ in range(5)]
    record = next(
        item
        for item in payloads[-1]["records"]
        if item["mode"] == "strong" and item["world_size"] == 2
    )
    record["global_samples_per_second"] *= 0.93
    result = AGGREGATOR.aggregate_sources(payloads)
    assert result["measurement_quality"]["status"] == "warning"
    assert result["measurement_quality"]["severely_unstable_cases"] == []


def test_dirty_or_inconsistent_sources_are_rejected() -> None:
    payloads = [source_payload() for _ in range(5)]
    payloads[2] = deepcopy(payloads[2])
    payloads[2]["git_dirty"] = True
    with pytest.raises(ValueError, match="dirty Git worktree"):
        AGGREGATOR.aggregate_sources(payloads)


def test_campaign_command_reuses_module03_runner_and_long_config() -> None:
    config = MODULE_ROOT / "configs" / "ddp_scaling_long.toml"
    command = CAMPAIGN.scaling_command(config, Path("run1.json"), 1800)
    runner = Path(command[1])
    assert runner.name == "run_scaling.py"
    assert runner.parent.name == "benchmarks"
    assert runner.parent.parent.name == "03_distributed_training"
    assert command[command.index("--config") + 1] == str(config)
    assert command[command.index("--timeout-seconds") + 1] == "1800"


def test_campaign_validation_rejects_the_old_short_window() -> None:
    config = load_benchmark_config(MODULE_ROOT / "configs" / "ddp_scaling_long.toml")
    CAMPAIGN.validate_campaign_config(config)
    config["measured_steps"] = 20
    with pytest.raises(ValueError, match="5000 measured"):
        CAMPAIGN.validate_campaign_config(config)
