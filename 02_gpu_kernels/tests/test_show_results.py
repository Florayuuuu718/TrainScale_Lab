from __future__ import annotations

import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARK_ROOT))

from show_results import summarize_payload  # noqa: E402


def test_beginner_summary_prints_latency_and_speedup() -> None:
    lines = summarize_payload(
        {
            "scope": "forward",
            "environment": {"gpu": "Teaching GPU"},
            "all_cases_passed": True,
            "comparisons": [
                {
                    "case_id": "vector_add_n257",
                    "pytorch_median_us": 10.0,
                    "triton_median_us": 8.0,
                    "triton_speedup_over_pytorch": 1.25,
                }
            ],
        }
    )
    output = "\n".join(lines)
    assert "Teaching GPU" in output
    assert "vector_add_n257" in output
    assert "1.250" in output


def test_cuda_summary_keeps_implementations_used_by_later_cases() -> None:
    lines = summarize_payload(
        {
            "comparisons": [
                {"case_id": "vector", "median_us": {"pytorch": 10.0, "triton": 9.0}},
                {
                    "case_id": "softmax",
                    "median_us": {
                        "pytorch": 10.0,
                        "triton_baseline": 12.0,
                        "triton": 9.0,
                    },
                },
            ]
        }
    )
    assert "triton_baseline_us" in "\n".join(lines)
