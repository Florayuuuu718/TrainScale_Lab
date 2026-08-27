from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = MODULE_ROOT / "benchmarks" / "build_nccl_tests.py"
SPEC = importlib.util.spec_from_file_location("module04_build", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_is_pinned_to_exact_official_commit() -> None:
    assert MODULE.NCCL_TESTS_REPOSITORY == "https://github.com/NVIDIA/nccl-tests.git"
    assert MODULE.NCCL_TESTS_TAG == "v2.19.7"
    assert MODULE.NCCL_TESTS_COMMIT == "1a65d7f0514b8da6a61ae235d1c5f38549478e29"
    command = MODULE.clone_command(Path("/tmp/nccl-tests"))
    assert command[command.index("--branch") + 1] == "v2.19.7"
    assert MODULE.build_command(Path("/tmp/nccl-tests"), 8)[-1] == "-j8"
