from __future__ import annotations

import json
from pathlib import Path

RESULT_ROOT = Path(__file__).resolve().parents[1] / "results"


def load(name: str) -> dict[str, object]:
    return json.loads((RESULT_ROOT / name).read_text(encoding="utf-8"))


def test_formal_result_artifacts_pass_their_correctness_gates() -> None:
    forward = load("triton_comparison_sm120_cu129.json")
    cuda = load("cuda_triton_comparison_sm120_cu129_cuda130.json")
    layer_norm = load("layer_norm_training_sm120_cu129.json")
    matmul = load("matmul_autotune_sm120_cu129.json")

    assert forward["all_cases_passed"] is True
    assert cuda["all_cases_passed"] is True
    assert layer_norm["all_cases_passed"] is True
    assert matmul["all_candidates_passed"] is True
    assert len(cuda["results"]) == 41
    assert len(layer_norm["comparisons"]) == 8
    assert len(matmul["selections"]) == 2


def test_compact_summary_references_content_hashes() -> None:
    summary = load("module02_summary_sm120.json")
    assert summary["all_required_results_passed"] is True
    assert summary["counts"]["cuda_triton_variant_successful_paths"] == 41  # type: ignore[index]
    for artifact in summary["source_artifacts"].values():  # type: ignore[union-attr]
        assert len(artifact["sha256"]) == 64


def test_release_acceptance_records_cpu_and_real_gpu_gates() -> None:
    acceptance = load("module02_acceptance_sm120.json")
    assert acceptance["status"] == "passed"
    assert acceptance["formal_results"]["all_required_results_passed"] is True  # type: ignore[index]
    assert acceptance["windows_cpu_validation"]["pytest"]["passed"] == 26  # type: ignore[index]
    assert acceptance["wsl_gpu_validation"]["gpu_pytest"]["passed"] == 15  # type: ignore[index]
    assert (
        acceptance["wsl_gpu_validation"]["beginner_operator_selection"][  # type: ignore[index]
            "successful_paths"
        ]
        == 4
    )
    assert (
        acceptance["wsl_gpu_validation"]["environment_probe"]["checks"][  # type: ignore[index]
            "cuda_toolkit_compile_and_run"
        ]
        is True
    )
