from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = MODULE_ROOT / "benchmarks" / "aggregate_runs.py"
SPEC = importlib.util.spec_from_file_location("module04_aggregate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source(time_us: float, busbw: float) -> dict[str, Any]:
    return {
        "artifact_type": "module04.nccl_tests",
        "status": "success",
        "git": {"commit": "abc", "dirty": False},
        "config_sha256": "config",
        "environment": {"timestamp": "different", "gpu_names": ["GPU0", "GPU1"]},
        "metrics": {
            "records": [
                {
                    "status": "success",
                    "case": {"id": "all_reduce_pair01"},
                    "rows": [
                        {
                            "size_bytes": 1024,
                            "out_of_place": {
                                "time_us": time_us,
                                "algbw_gbps": 10.0,
                                "busbw_gbps": busbw,
                                "wrong": 0,
                            },
                            "in_place": {
                                "time_us": time_us + 1,
                                "algbw_gbps": 9.0,
                                "busbw_gbps": busbw - 1,
                                "wrong": 0,
                            },
                        }
                    ],
                }
            ]
        },
    }


def test_aggregate_uses_median_and_requires_identical_rows() -> None:
    rows = MODULE.aggregate([source(30.0, 15.0), source(10.0, 13.0), source(20.0, 14.0)])
    out = next(row for row in rows if row["placement"] == "out_of_place")
    assert out["median_time_us"] == 20.0
    assert out["median_busbw_gbps"] == 14.0

    mismatched = source(40.0, 16.0)
    mismatched["metrics"]["records"][0]["rows"][0]["size_bytes"] = 2048
    with pytest.raises(ValueError, match="identical"):
        MODULE.aggregate([source(10.0, 13.0), source(20.0, 14.0), mismatched])


def test_aggregate_rejects_dirty_or_failed_sources() -> None:
    dirty = deepcopy(source(10.0, 13.0))
    dirty["git"]["dirty"] = True
    with pytest.raises(ValueError, match="clean worktree"):
        MODULE.aggregate([source(10.0, 13.0), source(20.0, 14.0), dirty])

