from __future__ import annotations

import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_nccl.contract import (  # noqa: E402
    expected_bus_bandwidth,
    load_bridge_config,
    load_cases,
    module03_mlp_parameter_count,
    nccl_test_command,
)


def test_repository_configs_load_and_cover_bridge_sizes() -> None:
    smoke = load_cases(MODULE_ROOT / "configs" / "nccl_smoke.toml")
    formal = load_cases(MODULE_ROOT / "configs" / "nccl_formal.toml")
    assert {case.collective for case in smoke} == {
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "broadcast",
    }
    assert any(case.min_bytes <= 10 * 1024**2 <= case.max_bytes for case in formal)
    assert {case.world_size for case in formal} == {2, 4}


def test_command_binds_visible_world_size_and_measurement_controls() -> None:
    case = load_cases(MODULE_ROOT / "configs" / "nccl_smoke.toml")[0]
    command = nccl_test_command(Path("/opt/nccl-tests/build"), case)
    assert command[0].endswith("all_reduce_perf")
    assert command[command.index("-g") + 1] == "2"
    assert command[command.index("-w") + 1] == "2"
    assert command[command.index("-n") + 1] == "5"


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config = tmp_path / "invalid.toml"
    config.write_text(
        """
[[case]]
id = "bad"
collective = "all_reduce"
devices = [0, 1]
min_bytes = 8
max_bytes = 16
step_factor = 2
warmup_iterations = 1
measured_iterations = 1
dtype = "float"
surprise = true
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown"):
        load_cases(config)


@pytest.mark.parametrize(
    ("collective", "expected"),
    [
        ("all_reduce", 30.0),
        ("all_gather", 15.0),
        ("reduce_scatter", 15.0),
        ("broadcast", 20.0),
    ],
)
def test_bus_bandwidth_normalization(collective: str, expected: float) -> None:
    assert expected_bus_bandwidth(20.0, collective, 4) == pytest.approx(expected)


def test_ddp_bridge_reuses_module03_workload_and_derives_payload() -> None:
    config = load_bridge_config(MODULE_ROOT / "configs" / "ddp_bridge.toml")
    count = module03_mlp_parameter_count(
        config["input_dim"], config["hidden_dim"], config["num_classes"]
    )
    assert count == 2_623_744
    assert count * 4 == 10_494_976
    assert config["world_sizes"] == [2, 4]
