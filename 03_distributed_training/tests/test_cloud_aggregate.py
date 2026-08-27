from __future__ import annotations

import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = MODULE_ROOT / "results" / "evidence" / "cloud_4x4090d"
sys.path.insert(0, str(MODULE_ROOT / "benchmarks"))

from aggregate_scaling_runs import aggregate, topology_summary  # noqa: E402

ARCHIVE_SHA256 = "63b0bb1efc17313cfd9df381afe67281d9daa2eb634a4fe570861ca7f3077e54"


def test_cloud_evidence_aggregates_to_six_real_cases() -> None:
    payload = aggregate(EVIDENCE_ROOT, ARCHIVE_SHA256)
    records = payload["records"]
    successful = [record for record in records if record["status"] == "success"]
    unavailable = [record for record in records if record["status"] == "unavailable"]

    assert payload["all_executable_cases_passed"] is True
    assert payload["source_git_commit"] == "d2b2882270a7e5ad8de6f666572497d2b3921703"
    assert payload["environment"]["cuda_device_count"] == 4
    assert len(successful) == 6
    assert len(unavailable) == 2
    assert all(record["repeat_count"] == 3 for record in records)
    assert all("global_samples_per_second" not in record for record in unavailable)

    strong_one = next(
        record
        for record in successful
        if record["mode"] == "strong" and record["world_size"] == 1
    )
    weak_four = next(
        record
        for record in successful
        if record["mode"] == "weak" and record["world_size"] == 4
    )
    assert strong_one["global_samples_per_second"] == pytest.approx(252630.9245693165)
    assert weak_four["speedup_over_1"] == pytest.approx(2.017635770733308)
    assert weak_four["scaling_efficiency"] == pytest.approx(0.504408942683327)


def test_cloud_topology_has_two_numa_domains_and_no_nvlink() -> None:
    topology = topology_summary((EVIDENCE_ROOT / "gpu-topology.txt").read_text(encoding="utf-8"))
    assert topology["gpu_row_count"] == 4
    assert topology["numa_affinities"] == ["0", "1"]
    assert topology["contains_cross_numa_sys_path"] is True
    assert topology["contains_nvlink"] is False
