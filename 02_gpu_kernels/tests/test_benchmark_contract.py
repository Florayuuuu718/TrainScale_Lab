from __future__ import annotations

import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from benchmark_contract import (  # noqa: E402
    load_case_ids,
    load_cases,
    load_matmul_candidates,
    percentile,
    select_case_ids,
    validate_result_record,
)


def test_checked_in_configs_are_valid_and_reusable() -> None:
    full = load_cases(MODULE_ROOT / "configs" / "benchmark_full.toml")
    smoke = load_case_ids(MODULE_ROOT / "configs" / "benchmark_smoke.toml", full)
    correctness = load_cases(MODULE_ROOT / "configs" / "correctness.toml")
    profiler = load_case_ids(MODULE_ROOT / "configs" / "profiler.toml", full)
    layer_norm = load_cases(MODULE_ROOT / "configs" / "layer_norm_training.toml")
    matmul_cases = load_cases(MODULE_ROOT / "configs" / "matmul_autotune_cases.toml")
    matmul_candidates = load_matmul_candidates(
        MODULE_ROOT / "configs" / "matmul_candidates.toml"
    )

    assert len(full) == 14
    assert set(smoke) <= set(full)
    assert set(profiler) <= set(full)
    assert len(layer_norm) == 4
    assert len(matmul_cases) == 2
    assert len(matmul_candidates) == 4
    assert {case["operator"] for case in correctness.values()} >= {
        "vector_add",
        "softmax",
        "layer_norm_backward",
        "matmul",
        "attention",
    }


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config = tmp_path / "bad.toml"
    config.write_text(
        """
[[case]]
id = "bad"
operator = "vector_add"
shape = [17]
dtype = "float32"
layout = "contiguous"
inner = 1
surprise = true
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown case fields"):
        load_cases(config)


def test_percentile_is_deterministic_and_validates_input() -> None:
    values = [9.0, 1.0, 5.0, 3.0, 7.0]
    assert percentile(values, 0.1) == 1.0
    assert percentile(values, 0.5) == 5.0
    assert percentile(values, 0.9) == 9.0
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 0.5)


def test_result_schema_requires_correctness_before_performance() -> None:
    record = {
        "case_id": "vector_add_n257",
        "implementation": "triton",
        "status": "success",
        "correctness": {"status": "passed"},
        "steady_state": {"latency_us": {"median": 10.0, "p10": 9.0, "p90": 12.0}},
    }
    validate_result_record(record)
    record["correctness"] = {"status": "failed"}
    with pytest.raises(ValueError, match="correctness gate"):
        validate_result_record(record)


def test_matmul_candidates_reject_invalid_warp_count(tmp_path: Path) -> None:
    config = tmp_path / "bad-candidate.toml"
    config.write_text(
        """
[[candidate]]
id = "bad"
block_m = 32
block_n = 32
block_k = 32
group_m = 8
num_warps = 3
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="num_warps"):
        load_matmul_candidates(config)


def test_beginner_case_selection_preserves_config_order() -> None:
    full = load_cases(MODULE_ROOT / "configs" / "benchmark_full.toml")
    smoke = load_case_ids(MODULE_ROOT / "configs" / "benchmark_smoke.toml", full)

    assert select_case_ids(full, smoke, operators=("softmax",)) == (
        "softmax_32x127",
        "softmax_256x1024",
    )
    assert select_case_ids(
        full,
        smoke,
        requested_case_ids=("matmul_509x509x509",),
    ) == ("matmul_509x509x509",)
    with pytest.raises(ValueError, match="not both"):
        select_case_ids(
            full,
            smoke,
            operators=("softmax",),
            requested_case_ids=("softmax_32x127",),
        )
