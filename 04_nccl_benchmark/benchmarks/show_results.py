"""Print a compact table from a module 04 nccl-tests artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _best_row(record: dict[str, Any]) -> dict[str, Any] | None:
    rows = record.get("rows", [])
    return max(rows, key=lambda row: row["out_of_place"]["busbw_gbps"], default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    print(f"artifact_type={payload['artifact_type']} status={payload['status']}")
    print("case | collective | GPUs | status | peak-size | peak-busbw")
    print("-" * 76)
    for record in payload["metrics"]["records"]:
        case = record["case"]
        best = _best_row(record)
        size = "-" if best is None else str(best["size_bytes"])
        busbw = "-" if best is None else f"{best['out_of_place']['busbw_gbps']:.2f} GB/s"
        print(
            f"{case['id']} | {case['collective']} | {case['world_size']} | "
            f"{record['status']} | {size} | {busbw}"
        )
        if record["status"] == "unavailable":
            print(f"  reason: {record['reason']}")
    print(f"boundary={payload['boundary']}")


if __name__ == "__main__":
    main()

