from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_engine.engine import train_step  # noqa: E402
from trainscale_engine.reducer import BulkReducer  # noqa: E402


def test_optimizer_step_requires_a_ddp_model_when_no_manual_reducer() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="reserved for"):
        train_step(
            model,
            optimizer,
            torch.zeros((2, 2), dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            reducer=None,
        )


def test_reducer_rejects_optimizer_boundary_before_finish() -> None:
    reducer = BulkReducer(nn.Linear(2, 2))
    reducer.begin_backward(sync=True)
    with pytest.raises(RuntimeError, match="forbidden"):
        reducer.assert_complete()
