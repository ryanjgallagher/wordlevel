import runpy
from pathlib import Path

import pytest


SNIPPETS_DIR = Path(__file__).resolve().parents[2] / "docs" / "snippets"


@pytest.mark.parametrize(
    "path",
    sorted(SNIPPETS_DIR.rglob("*.py")),
    ids=lambda p: p.relative_to(SNIPPETS_DIR).as_posix(),
)
def test_snippet_runs(path: Path) -> None:
    """
    Tests that all code snippets for the documentation can be run end-to-end without errors
    """
    runpy.run_path(str(path), run_name="__main__")
