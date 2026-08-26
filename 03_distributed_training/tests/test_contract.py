from __future__ import annotations

import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_distributed.contract import (  # noqa: E402
    add_scaling_metrics,
    analyze_sampler_shards,
    load_benchmark_config,
    load_correctness_config,
)


def test_checked_in_configs_are_valid() -> None:
    correctness = load_correctness_config(MODULE_ROOT / "configs" / "correctness.toml")
    cpu_smoke = load_benchmark_config(MODULE_ROOT / "configs" / "cpu_scaling_smoke.toml")
    cpu = load_benchmark_config(MODULE_ROOT / "configs" / "cpu_scaling.toml")
    gpu_smoke = load_benchmark_config(MODULE_ROOT / "configs" / "gpu_scaling_smoke.toml")
    gpu = load_benchmark_config(MODULE_ROOT / "configs" / "gpu_scaling.toml")
    assert correctness["global_batch_size"] == 64
    assert cpu_smoke["world_sizes"] == [1, 2]
    assert cpu["world_sizes"] == [1, 2, 4]
    assert gpu_smoke["measured_steps"] == 3
    assert (gpu["device"], gpu["backend"]) == ("cuda", "nccl")


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    source = (MODULE_ROOT / "configs" / "correctness.toml").read_text(encoding="utf-8")
    path.write_text(source + "surprise = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_correctness_config(path)


def test_sampler_analysis_distinguishes_coverage_from_padding() -> None:
    analysis = analyze_sampler_shards([[0, 2, 4], [1, 3, 0]], dataset_size=5)
    assert analysis["coverage_complete"] is True
    assert analysis["padding_duplicates"] == 1
    assert analysis["missing_samples"] == []


def test_scaling_metrics_use_world_one_baseline() -> None:
    records = [
        {
            "status": "success",
            "mode": "strong",
            "world_size": 1,
            "global_samples_per_second": 100.0,
        },
        {
            "status": "success",
            "mode": "strong",
            "world_size": 2,
            "global_samples_per_second": 180.0,
        },
        {"status": "unavailable", "mode": "strong", "world_size": 4},
    ]
    add_scaling_metrics(records)
    assert records[1]["speedup_over_1"] == 1.8
    assert records[1]["scaling_efficiency"] == 0.9
    assert "speedup_over_1" not in records[2]
