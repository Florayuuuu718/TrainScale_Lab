from __future__ import annotations

import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_nccl.parser import correctness_passed, parse_nccl_tests_output  # noqa: E402

SAMPLE = """
# nccl-tests mock excerpt
# size count type redop root out-of-place in-place
8 2 float sum -1 12.50 0.01 0.01 0 11.00 0.01 0.01 0
10485760 2621440 float sum -1 250.00 41.94 62.91 0 245.00 42.80 64.20 0
# Out of bounds values : 0 OK
"""


def test_parse_rows_and_correctness() -> None:
    rows = parse_nccl_tests_output(SAMPLE)
    assert [row["size_bytes"] for row in rows] == [8, 10485760]
    assert rows[1]["out_of_place"]["busbw_gbps"] == pytest.approx(62.91)
    assert correctness_passed(rows)


def test_nonzero_wrong_count_fails_correctness() -> None:
    rows = parse_nccl_tests_output(SAMPLE.replace("64.20 0", "64.20 2"))
    assert not correctness_passed(rows)


def test_parser_rejects_stdout_without_data_rows() -> None:
    with pytest.raises(ValueError, match="parseable"):
        parse_nccl_tests_output("# only headers\n")

