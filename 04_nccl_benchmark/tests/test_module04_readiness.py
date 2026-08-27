from __future__ import annotations

import json
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]


def test_local_readiness_does_not_claim_multi_gpu_results() -> None:
    path = MODULE_ROOT / "results" / "module04_local_readiness.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed_local_pre_multi_gpu_gates"
    assert payload["wsl_validation"]["environment"]["gpu_count"] == 1
    assert payload["wsl_validation"]["collective_smoke"]["artifact_status"] == "unavailable"
    assert "module04 summary and final acceptance" in payload["pending_multi_gpu_gates"]
    assert "no multi-GPU performance measurements" in payload["boundary"]

