from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

from artifact_contract import (  # noqa: E402
    build_artifact,
    canonical_sha256,
    percentile,
    validate_artifact,
)


def test_percentile_and_canonical_hash_are_deterministic() -> None:
    assert percentile([9.0, 1.0, 5.0], 0.5) == 5.0
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_successful_artifact_requires_passed_correctness() -> None:
    config = {"case": "small"}
    payload = {
        "schema_version": 1,
        "artifact_type": "test",
        "generated_at": "2026-08-27T00:00:00+08:00",
        "git": {"commit": "abc", "dirty": False},
        "environment": {},
        "config": config,
        "config_sha256": canonical_sha256(config),
        "measurement": {},
        "status": "success",
        "correctness": {"status": "not_run"},
        "metrics": {},
        "raw_artifacts": [],
        "boundary": "test only",
    }
    with pytest.raises(ValueError, match="passed correctness"):
        validate_artifact(payload)


def test_build_unavailable_artifact_records_clean_status_contract() -> None:
    payload = build_artifact(
        artifact_type="test.unavailable",
        repository_root=ROOT,
        environment={},
        config={"required_gpus": 2},
        measurement={},
        status="unavailable",
        correctness={"status": "not_run"},
        metrics={},
        raw_artifacts=[],
        boundary="no multi-GPU hardware",
    )
    assert payload["status"] == "unavailable"
    assert len(payload["config_sha256"]) == 64

