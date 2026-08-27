"""Validate three module 04 formal runs and aggregate row metrics by median."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks"))

from artifact_contract import (  # noqa: E402
    build_artifact,
    canonical_sha256,
    percentile,
    sha256_file,
)


def _stable_environment(environment: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in environment.items() if key != "timestamp"}


def validate_sources(sources: list[dict[str, Any]]) -> None:
    if len(sources) != 3:
        raise ValueError("formal aggregation requires exactly three runs")
    if any(source.get("artifact_type") != "module04.nccl_tests" for source in sources):
        raise ValueError("all sources must be module04.nccl_tests artifacts")
    if any(source.get("status") != "success" for source in sources):
        raise ValueError("all source runs must have status=success")
    if any(source.get("git", {}).get("dirty") for source in sources):
        raise ValueError("formal source runs must use a clean worktree")
    commits = {source["git"]["commit"] for source in sources}
    configs = {source["config_sha256"] for source in sources}
    environments = {
        canonical_sha256(_stable_environment(source["environment"])) for source in sources
    }
    if len(commits) != 1 or len(configs) != 1 or len(environments) != 1:
        raise ValueError("formal runs must share commit, config, and stable environment")


def _row_index(source: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    index: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in source["metrics"]["records"]:
        if record["status"] != "success":
            raise ValueError("formal source contains a non-success case")
        case_id = record["case"]["id"]
        for row in record["rows"]:
            for placement in ("out_of_place", "in_place"):
                key = (case_id, row["size_bytes"], placement)
                if key in index:
                    raise ValueError(f"duplicate row key {key}")
                index[key] = row[placement]
    return index


def aggregate(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_sources(sources)
    indexes = [_row_index(source) for source in sources]
    keys = set(indexes[0])
    if any(set(index) != keys for index in indexes[1:]):
        raise ValueError("formal runs do not contain identical case/size/placement rows")
    rows: list[dict[str, Any]] = []
    for case_id, size_bytes, placement in sorted(keys):
        metrics = [index[(case_id, size_bytes, placement)] for index in indexes]
        if any(metric["wrong"] not in {0, None} for metric in metrics):
            raise ValueError("cannot aggregate rows with correctness errors")
        rows.append(
            {
                "case_id": case_id,
                "size_bytes": size_bytes,
                "placement": placement,
                "median_time_us": percentile([m["time_us"] for m in metrics], 0.5),
                "median_algbw_gbps": percentile([m["algbw_gbps"] for m in metrics], 0.5),
                "median_busbw_gbps": percentile([m["busbw_gbps"] for m in metrics], 0.5),
                "raw_repetitions": metrics,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs=3, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [json.loads(path.read_text(encoding="utf-8")) for path in args.sources]
    rows = aggregate(sources)
    raw_sources = [
        {"path": str(path), "sha256": sha256_file(path)} for path in args.sources
    ]
    payload = build_artifact(
        artifact_type="module04.nccl_tests.aggregate",
        repository_root=REPOSITORY_ROOT,
        environment=sources[0]["environment"],
        config=sources[0]["config"],
        measurement={"formal_run_count": 3, "aggregation": "median of matching rows"},
        status="success",
        correctness={"status": "passed"},
        metrics={"rows": rows, "row_count": len(rows)},
        raw_artifacts=raw_sources,
        boundary=sources[0]["boundary"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
