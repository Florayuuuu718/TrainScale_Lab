from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from trainscale_collective.algorithms import PendingExchange  # noqa: E402


class Request:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


def test_pending_exchange_waits_every_handle_exactly_once() -> None:
    requests = [Request(), Request()]
    received = torch.tensor([1.0])
    pending = PendingExchange(requests, received)
    assert pending.wait() is received
    assert all(request.waited for request in requests)
    with pytest.raises(RuntimeError, match="only be waited once"):
        pending.wait()
