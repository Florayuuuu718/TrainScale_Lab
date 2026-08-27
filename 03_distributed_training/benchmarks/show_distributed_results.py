"""Print a compact beginner-readable table for module 03 result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def number(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def render(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> list[str]:
    text_rows = [[number(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in text_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    return [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
        *[
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            for row in text_rows
        ],
    ]


def summarize(payload: dict[str, Any]) -> list[str]:
    lines = [f"scope: {payload.get('scope', 'unknown')}"]
    if "records" in payload:
        rows = [
            (
                record["mode"],
                record["world_size"],
                record["status"],
                record.get("global_batch_size", "-"),
                record.get("global_samples_per_second", "-"),
                record.get("speedup_over_1", "-"),
                record.get("scaling_efficiency", "-"),
            )
            for record in payload["records"]
        ]
        lines.extend(
            render(
                ("mode", "world", "status", "global_batch", "samples/s", "speedup", "efficiency"),
                rows,
            )
        )
        return lines
    if "result" in payload:
        result = payload["result"]
        lines.append(f"status: {result['status']}")
        for key in (
            "world_size",
            "set_epoch_changed_order",
            "rank_parameter_max_difference",
            "gradient_max_error_vs_global_batch",
            "parameter_max_error_vs_global_batch",
            "continuous_vs_resumed_parameter_max_error",
            "resume_start_epoch",
        ):
            if key in result:
                lines.append(f"{key}: {number(result[key])}")
        if "epochs" in result:
            lines.extend(
                render(
                    ("epoch", "coverage", "padding_duplicates", "samples_per_rank"),
                    [
                        (
                            epoch["epoch"],
                            epoch["analysis"]["coverage_complete"],
                            epoch["analysis"]["padding_duplicates"],
                            epoch["analysis"]["samples_per_rank"],
                        )
                        for epoch in result["epochs"]
                    ],
                )
            )
        return lines
    if "ranks" in payload:
        profile_rows: list[tuple[Any, ...]] = []
        for rank in payload["ranks"]:
            top = rank.get("distributed_rows", [{}])[0]
            profile_rows.append(
                (
                    rank["rank"],
                    len(rank.get("distributed_rows", [])),
                    top.get("name", "-"),
                    top.get("cpu_time_total_us", "-"),
                )
            )
        lines.extend(render(("rank", "comm_rows", "top_row", "cpu_total_us"), profile_rows))
        return lines
    raise ValueError("unsupported module 03 result schema")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    print("\n".join(summarize(payload)))


if __name__ == "__main__":
    main()
