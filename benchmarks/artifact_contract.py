"""Shared provenance and status contract for module 04-07 result artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

STATUSES = {"success", "failed", "unavailable"}
CORRECTNESS_STATUSES = {"passed", "failed", "not_run"}


def percentile(values: list[float], fraction: float) -> float:
    """Return the repository-wide nearest-index percentile sample."""
    if not values:
        raise ValueError("percentile requires at least one sample")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def sha256_file(path: Path) -> str:
    """Hash a file without loading a potentially large trace into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible configuration using a stable representation."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_state(repository_root: Path) -> dict[str, Any]:
    """Collect the exact repository revision used by an experiment."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository_root, text=True
        ).strip()
    )
    return {"commit": commit, "dirty": dirty}


def validate_artifact(payload: dict[str, Any]) -> None:
    """Validate the fields and cross-field rules shared by future modules."""
    required = {
        "schema_version",
        "artifact_type",
        "generated_at",
        "git",
        "environment",
        "config",
        "config_sha256",
        "measurement",
        "status",
        "correctness",
        "metrics",
        "raw_artifacts",
        "boundary",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"artifact is missing fields: {sorted(missing)}")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported shared artifact schema")
    if not isinstance(payload["artifact_type"], str) or not payload["artifact_type"]:
        raise ValueError("artifact_type must be a non-empty string")
    if payload["status"] not in STATUSES:
        raise ValueError(f"unsupported artifact status {payload['status']!r}")
    correctness = payload["correctness"]
    if not isinstance(correctness, dict) or correctness.get("status") not in CORRECTNESS_STATUSES:
        raise ValueError("correctness.status must be passed, failed, or not_run")
    if payload["status"] == "success" and correctness["status"] != "passed":
        raise ValueError("successful performance artifacts require passed correctness")
    if payload["status"] == "unavailable" and correctness["status"] != "not_run":
        raise ValueError("unavailable artifacts require correctness.status=not_run")
    if payload["config_sha256"] != canonical_sha256(payload["config"]):
        raise ValueError("config_sha256 does not match config")
    if not isinstance(payload["boundary"], str) or not payload["boundary"]:
        raise ValueError("boundary must be a non-empty string")


def build_artifact(
    *,
    artifact_type: str,
    repository_root: Path,
    environment: dict[str, Any],
    config: dict[str, Any],
    measurement: dict[str, Any],
    status: str,
    correctness: dict[str, Any],
    metrics: dict[str, Any],
    raw_artifacts: list[dict[str, Any]],
    boundary: str,
) -> dict[str, Any]:
    """Create and validate a shared artifact envelope."""
    payload = {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "generated_at": datetime.now().astimezone().isoformat(),
        "git": git_state(repository_root),
        "environment": environment,
        "config": config,
        "config_sha256": canonical_sha256(config),
        "measurement": measurement,
        "status": status,
        "correctness": correctness,
        "metrics": metrics,
        "raw_artifacts": raw_artifacts,
        "boundary": boundary,
    }
    validate_artifact(payload)
    return payload

