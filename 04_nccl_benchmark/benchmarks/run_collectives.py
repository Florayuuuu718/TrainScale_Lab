"""Run configured nccl-tests cases or record why multi-GPU cases are unavailable."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks"))

from artifact_contract import build_artifact, sha256_file  # noqa: E402
from trainscale_nccl.contract import NcclCase, load_cases, nccl_test_command  # noqa: E402
from trainscale_nccl.environment import collect_environment  # noqa: E402
from trainscale_nccl.parser import correctness_passed, parse_nccl_tests_output  # noqa: E402


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _unavailable_reason(case: NcclCase, binary_directory: Path, available_gpus: int) -> str | None:
    if platform.system() != "Linux":
        return "nccl-tests execution requires Linux/WSL"
    if available_gpus < case.world_size:
        return f"case requires {case.world_size} GPUs but only {available_gpus} are visible"
    binary = binary_directory / case.binary
    if not binary.is_file():
        return f"missing pinned nccl-tests binary: {binary}"
    return None


def _run_case(
    case: NcclCase,
    *,
    binary_directory: Path,
    raw_directory: Path,
    available_gpus: int,
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    command = nccl_test_command(binary_directory, case)
    reason = _unavailable_reason(case, binary_directory, available_gpus)
    if reason is not None:
        return (
            {
                "case": case.to_dict(),
                "status": "unavailable",
                "command": command,
                "reason": reason,
            },
            None,
        )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in case.devices)
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return (
            {
                "case": case.to_dict(),
                "status": "failed",
                "command": command,
                "reason": f"timed out after {timeout_seconds} seconds",
                "stdout_tail": (error.stdout or "")[-4000:],
                "stderr_tail": (error.stderr or "")[-4000:],
            },
            None,
        )
    raw_directory.mkdir(parents=True, exist_ok=True)
    stdout_path = raw_directory / f"{case.id}.stdout.txt"
    stderr_path = raw_directory / f"{case.id}.stderr.txt"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    raw_artifact = {
        "case_id": case.id,
        "stdout": _relative(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr": _relative(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
    }
    if completed.returncode != 0:
        return (
            {
                "case": case.to_dict(),
                "status": "failed",
                "command": command,
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-4000:],
            },
            raw_artifact,
        )
    try:
        rows = parse_nccl_tests_output(completed.stdout)
    except ValueError as error:
        return (
            {
                "case": case.to_dict(),
                "status": "failed",
                "command": command,
                "returncode": completed.returncode,
                "reason": str(error),
            },
            raw_artifact,
        )
    passed = correctness_passed(rows)
    return (
        {
            "case": case.to_dict(),
            "status": "success" if passed else "failed",
            "command": command,
            "returncode": completed.returncode,
            "correctness": {"status": "passed" if passed else "failed"},
            "rows": rows,
        },
        raw_artifact,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--binary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    cases = load_cases(args.config)
    environment = collect_environment()
    raw_directory = args.raw_directory or args.output.parent / "raw" / args.output.stem
    records: list[dict[str, Any]] = []
    raw_artifacts: list[dict[str, Any]] = []
    for case in cases:
        record, raw = _run_case(
            case,
            binary_directory=args.binary_directory,
            raw_directory=raw_directory,
            available_gpus=environment["cuda_device_count"],
            timeout_seconds=args.timeout_seconds,
        )
        records.append(record)
        if raw is not None:
            raw_artifacts.append(raw)
        print(f"{case.id}: {record['status']}")
    if any(record["status"] == "failed" for record in records):
        status = "failed"
        correctness = {"status": "failed"}
    elif any(record["status"] == "success" for record in records):
        status = "success"
        correctness = {"status": "passed"}
    else:
        status = "unavailable"
        correctness = {"status": "not_run"}
    payload = build_artifact(
        artifact_type="module04.nccl_tests",
        repository_root=REPOSITORY_ROOT,
        environment=environment,
        config={"path": _relative(args.config), "cases": [case.to_dict() for case in cases]},
        measurement={
            "runner": "single-process multi-GPU nccl-tests",
            "timeout_seconds": args.timeout_seconds,
            "repeat_policy": "one artifact per invocation; formal acceptance aggregates three runs",
        },
        status=status,
        correctness=correctness,
        metrics={
            "records": records,
            "successful_case_count": sum(r["status"] == "success" for r in records),
            "failed_case_count": sum(r["status"] == "failed" for r in records),
            "unavailable_case_count": sum(r["status"] == "unavailable" for r in records),
        },
        raw_artifacts=raw_artifacts,
        boundary=(
            "Absolute bandwidth is comparable only within the recorded host, topology, and "
            "software environment. Unavailable cases contain no fabricated measurements."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(1 if status == "failed" else 0)


if __name__ == "__main__":
    main()

