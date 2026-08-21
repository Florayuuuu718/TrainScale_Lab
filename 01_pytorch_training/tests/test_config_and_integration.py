from pathlib import Path

import pytest
import torch
from trainscale_training.config import ExperimentConfig, load_config
from trainscale_training.engine import run_training


def test_config_rejects_amp_on_cpu() -> None:
    with pytest.raises(ValueError, match="AMP"):
        ExperimentConfig(precision="amp", device="cpu").validate()


def test_toml_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('experiment_name = "bad"\nunknown = 1\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config fields"):
        load_config(path)


def test_configured_training_writes_results_and_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "run"
    config = ExperimentConfig(
        experiment_name="integration",
        train_samples=32,
        valid_samples=16,
        epochs=2,
        batch_size=8,
        output_dir=str(output),
    )
    summary = run_training(config)
    assert len(summary["history"]) == 2
    assert summary["global_step"] == 8
    assert (output / "config.json").is_file()
    assert (output / "environment.json").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "summary.json").is_file()
    checkpoint = torch.load(output / "last.pt", weights_only=False)
    assert checkpoint["epoch"] == 2
    assert checkpoint["scheduler"] is not None
    assert checkpoint["scaler"] is not None
