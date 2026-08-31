from __future__ import annotations

import re
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.ipynb"))


def test_expected_notebooks_exist() -> None:
    assert [path.name[:2] for path in NOTEBOOKS] == [f"{number:02d}" for number in range(8)]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_structure_and_links(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    ids = [cell.id for cell in notebook.cells]
    assert all(ids)
    assert len(ids) == len(set(ids))
    text = "\n".join("".join(cell.source) for cell in notebook.cells)
    assert "MODE = \"reference\"" in text
    assert not re.search(r"[A-Za-z]:\\\\(?:Users|00A)\\\\", text)
    for target in re.findall(r"\[[^]]+\]\(([^)#]+)", text):
        if "://" not in target:
            assert (path.parent / target).resolve().exists(), f"broken link: {target}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_reference_mode_run_all(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAINSCALE_NOTEBOOK_MODE", "reference")
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
