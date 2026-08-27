"""Print a compact, beginner-friendly view of a module 02 result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def format_number(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{value:.3f}"
    return str(value)


def table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> list[str]:
    rendered = [[format_number(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    header_line = "  ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    )
    divider = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rendered
    ]
    return [header_line, divider, *body]


def comparison_lines(comparisons: list[dict[str, Any]]) -> list[str]:
    if not comparisons:
        return []
    first = comparisons[0]
    if "median_us" in first:
        implementations = tuple(
            dict.fromkeys(
                name
                for comparison in comparisons
                for name in comparison.get("median_us", {})
            )
        )
        rows = [
            (
                comparison["case_id"],
                *(comparison.get("median_us", {}).get(name, "-") for name in implementations),
            )
            for comparison in comparisons
        ]
        return table(("case", *(f"{name}_us" for name in implementations)), rows)
    phase_column = "phase" in first
    rows = [
        (
            comparison["case_id"],
            *((comparison.get("phase", "-"),) if phase_column else ()),
            comparison.get("pytorch_median_us", "-"),
            comparison.get("triton_median_us", "-"),
            comparison.get("triton_speedup_over_pytorch", "-"),
        )
        for comparison in comparisons
    ]
    headers = (
        "case",
        *(("phase",) if phase_column else ()),
        "pytorch_us",
        "triton_us",
        "speedup",
    )
    return table(headers, rows)


def selection_lines(selections: list[dict[str, Any]]) -> list[str]:
    rows = [
        (
            selection["case_id"],
            selection.get("selected_candidate", "-"),
            selection.get("pytorch_median_us", "-"),
            selection.get("selected_median_us", "-"),
            selection.get("selected_speedup_over_pytorch", "-"),
        )
        for selection in selections
    ]
    return table(
        ("case", "selected", "pytorch_us", "triton_us", "speedup"),
        rows,
    )


def profiler_lines(cases: dict[str, dict[str, Any]]) -> list[str]:
    rows: list[tuple[Any, ...]] = []
    for case_id, case in cases.items():
        device_rows = case.get("device_rows", [])
        top = max(device_rows, key=lambda row: row.get("device_time_total_us", 0), default={})
        rows.append(
            (
                case_id,
                case.get("iterations", "-"),
                top.get("device_time_total_us", "-"),
                top.get("name", "no device row")[:60],
            )
        )
    return table(("case", "iters", "top_device_us", "top_device_row"), rows)


def summarize_payload(payload: dict[str, Any]) -> list[str]:
    environment = payload.get("environment", {})
    all_passed = payload.get(
        "all_cases_passed", payload.get("all_candidates_passed", "n/a")
    )
    lines = [
        f"scope: {payload.get('scope', 'profiler')}",
        f"gpu: {environment.get('gpu', 'unknown')}",
        f"all passed: {all_passed}",
        "",
    ]
    if "comparisons" in payload:
        lines.extend(comparison_lines(payload["comparisons"]))
    elif "selections" in payload:
        lines.extend(selection_lines(payload["selections"]))
    elif isinstance(payload.get("cases"), dict):
        lines.extend(profiler_lines(payload["cases"]))
    else:
        raise ValueError("unsupported module 02 result shape")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Result JSON produced by a module 02 runner")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    print("\n".join(summarize_payload(payload)))


if __name__ == "__main__":
    main()
