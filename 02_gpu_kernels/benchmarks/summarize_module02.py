"""Build the compact module 02 acceptance summary from formal result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = MODULE_ROOT / "results"
SOURCES = {
    "environment": "sm120_environment_validation.json",
    "forward": "triton_comparison_sm120_cu129.json",
    "profiler": "triton_profiler_sm120_cu129.json",
    "cuda_comparison": "cuda_triton_comparison_sm120_cu129_cuda130.json",
    "layer_norm_training": "layer_norm_training_sm120_cu129.json",
    "matmul_autotune": "matmul_autotune_sm120_cu129.json",
}


def load_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    payloads: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    for name, filename in SOURCES.items():
        path = RESULT_ROOT / filename
        raw = path.read_bytes()
        payloads[name] = json.loads(raw)
        manifests[name] = {
            "path": f"02_gpu_kernels/results/{filename}",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return payloads, manifests


def build_summary() -> dict[str, Any]:
    payloads, manifests = load_sources()
    forward = payloads["forward"]
    cuda = payloads["cuda_comparison"]
    layer_norm = payloads["layer_norm_training"]
    matmul = payloads["matmul_autotune"]
    required_passes = {
        "forward_14_cases": forward["all_cases_passed"],
        "cuda_four_way_9_cases": cuda["all_cases_passed"],
        "layer_norm_forward_backward_8_comparisons": layer_norm["all_cases_passed"],
        "matmul_2_shapes_4_candidates": matmul["all_candidates_passed"],
    }
    cuda_winners = []
    for comparison in cuda["comparisons"]:
        medians = comparison["median_us"]
        winner = min(medians, key=medians.get)
        cuda_winners.append(
            {
                "case_id": comparison["case_id"],
                "fastest_implementation": winner,
                "median_us": medians[winner],
                "all_medians_us": medians,
            }
        )
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "scope": "Module 02 formal acceptance summary on the recorded SM 12.0 environment",
        "environment": payloads["environment"]["hardware"],
        "required_passes": required_passes,
        "all_required_results_passed": all(required_passes.values()),
        "counts": {
            "pytorch_triton_forward_cases": len(forward["comparisons"]),
            "cuda_four_way_cases": len(cuda["comparisons"]),
            "cuda_triton_variant_successful_paths": len(
                [result for result in cuda["results"] if result["status"] == "success"]
            ),
            "layer_norm_phase_comparisons": len(layer_norm["comparisons"]),
            "matmul_candidates_per_shape": len(matmul["candidates"]),
            "profiler_cases": len(payloads["profiler"]["cases"]),
        },
        "cuda_four_way_winners": cuda_winners,
        "layer_norm_comparisons": layer_norm["comparisons"],
        "matmul_selections": matmul["selections"],
        "source_artifacts": manifests,
        "boundary": (
            "These are single-machine educational results, not cross-hardware performance claims."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_ROOT / "module02_summary_sm120.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(f"all_required_results_passed={payload['all_required_results_passed']}")


if __name__ == "__main__":
    main()
