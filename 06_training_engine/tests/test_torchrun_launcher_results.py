from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks"))

from torchrun_launcher import rank_result_files  # noqa: E402


def test_rank_result_files_excludes_profiler_traces(tmp_path: Path) -> None:
    for name in (
        "rank_0.json",
        "rank_1.json",
        "rank_0_trace.json",
        "rank_1_trace.json",
        "rank_summary.json",
    ):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    assert [path.name for path in rank_result_files(tmp_path)] == [
        "rank_0.json",
        "rank_1.json",
    ]
