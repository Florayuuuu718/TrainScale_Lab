from __future__ import annotations

import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "benchmarks"))

from show_distributed_results import summarize  # noqa: E402


def test_scaling_summary_shows_unavailable_without_fake_throughput() -> None:
    output = "\n".join(
        summarize(
            {
                "scope": "cuda/nccl",
                "records": [
                    {
                        "mode": "strong",
                        "world_size": 2,
                        "status": "unavailable",
                    }
                ],
            }
        )
    )
    assert "unavailable" in output
    assert "strong" in output
