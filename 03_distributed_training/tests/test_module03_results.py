from __future__ import annotations

import json
from pathlib import Path

RESULT_ROOT = Path(__file__).resolve().parents[1] / "results"


def load(name: str) -> dict[str, object]:
    return json.loads((RESULT_ROOT / name).read_text(encoding="utf-8"))


def test_formal_correctness_and_profile_results_pass() -> None:
    for name in (
        "gloo_semantics_cpu.json",
        "sampler_sharding_cpu.json",
        "gradient_equivalence_cpu.json",
        "checkpoint_resume_cpu.json",
        "ddp_profile_cpu.json",
    ):
        assert load(name)["all_checks_passed"] is True


def test_scaling_results_separate_success_from_unavailable() -> None:
    cpu = load("scaling_cpu.json")
    gpu = load("scaling_nccl_sm120.json")
    cloud = load("scaling_nccl_4x4090d.json")
    assert cpu["all_executable_cases_passed"] is True
    assert gpu["all_executable_cases_passed"] is True
    assert cloud["all_executable_cases_passed"] is True
    cpu_records = cpu["records"]
    gpu_records = gpu["records"]
    assert sum(record["status"] == "success" for record in cpu_records) == 6  # type: ignore[union-attr]
    assert sum(record["status"] == "success" for record in gpu_records) == 2  # type: ignore[union-attr]
    assert sum(record["status"] == "unavailable" for record in gpu_records) == 6  # type: ignore[union-attr]
    for record in gpu_records:  # type: ignore[union-attr]
        if record["status"] == "unavailable":
            assert "global_samples_per_second" not in record
    cloud_records = cloud["records"]
    assert sum(record["status"] == "success" for record in cloud_records) == 6  # type: ignore[union-attr]
    assert sum(record["status"] == "unavailable" for record in cloud_records) == 2  # type: ignore[union-attr]
    assert all(record["repeat_count"] == 3 for record in cloud_records)  # type: ignore[union-attr]


def test_summary_content_addresses_every_formal_source() -> None:
    summary = load("module03_summary.json")
    acceptance = load("module03_acceptance.json")
    assert summary["all_executable_gates_passed"] is True
    assert len(summary["source_artifacts"]) == 9  # type: ignore[arg-type]
    for artifact in summary["source_artifacts"].values():  # type: ignore[union-attr]
        assert len(artifact["sha256"]) == 64
    assert acceptance["formal_results"]["all_executable_gates_passed"] is True  # type: ignore[index]
    assert acceptance["cloud_validation"]["successful_repetition_cases"] == 18  # type: ignore[index]
