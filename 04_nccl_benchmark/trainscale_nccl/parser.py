"""Parse the stable tabular portion of NVIDIA nccl-tests stdout."""

from __future__ import annotations

from typing import Any


def _error_count(token: str) -> int | None:
    if token.lower() in {"n/a", "na", "-"}:
        return None
    return int(token)


def parse_nccl_tests_output(stdout: str) -> list[dict[str, Any]]:
    """Extract out-of-place and in-place metrics from nccl-tests data rows."""
    records: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < 13:
            continue
        try:
            size_bytes = int(tokens[0])
            count = int(tokens[1])
            out_time, out_algbw, out_busbw = map(float, tokens[-8:-5])
            out_errors = _error_count(tokens[-5])
            in_time, in_algbw, in_busbw = map(float, tokens[-4:-1])
            in_errors = _error_count(tokens[-1])
        except ValueError:
            continue
        records.append(
            {
                "size_bytes": size_bytes,
                "element_count": count,
                "dtype": tokens[2],
                "out_of_place": {
                    "time_us": out_time,
                    "algbw_gbps": out_algbw,
                    "busbw_gbps": out_busbw,
                    "wrong": out_errors,
                },
                "in_place": {
                    "time_us": in_time,
                    "algbw_gbps": in_algbw,
                    "busbw_gbps": in_busbw,
                    "wrong": in_errors,
                },
            }
        )
    if not records:
        raise ValueError("nccl-tests stdout did not contain parseable data rows")
    return records


def correctness_passed(records: list[dict[str, Any]]) -> bool:
    """Require every available nccl-tests wrong-count to be zero."""
    if not records:
        return False
    values = [
        placement["wrong"]
        for record in records
        for placement in (record["out_of_place"], record["in_place"])
        if placement["wrong"] is not None
    ]
    return bool(values) and all(value == 0 for value in values)

