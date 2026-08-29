from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = REPOSITORY_ROOT / "06_training_engine"
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "01_pytorch_training"))

from trainscale_engine.checkpoint import (  # noqa: E402
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def _step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.tensor([[1.0, -1.0]])).square().mean()
    loss.backward()
    optimizer.step()


def test_checkpoint_resume_next_step_matches_continuous_training(tmp_path: Path) -> None:
    torch.manual_seed(7)
    continuous = nn.Linear(2, 2)
    continuous_optimizer = torch.optim.SGD(continuous.parameters(), lr=0.1, momentum=0.9)
    _step(continuous, continuous_optimizer)
    checkpoint = build_checkpoint(
        model=continuous,
        optimizer=continuous_optimizer,
        epoch=0,
        global_step=1,
    )
    path = tmp_path / "engine.pt"
    save_checkpoint(path, checkpoint)
    _step(continuous, continuous_optimizer)

    restored = nn.Linear(2, 2)
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9)
    state = load_checkpoint(path, model=restored, optimizer=restored_optimizer)
    assert state["global_step"] == 1
    _step(restored, restored_optimizer)
    for left, right in zip(continuous.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(left, right)
