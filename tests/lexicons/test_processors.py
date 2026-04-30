import importlib
import pathlib

import pytest

from lexicons.processor import LexiconProcessor


def _discover_processors() -> list[type[LexiconProcessor]]:
    """
    Automatically finds all implemented processors (subclasses of LexiconProcessor)
    """
    lexicons_dir = pathlib.Path(__file__).parents[2] / "lexicons"

    for process_file in sorted(lexicons_dir.glob("*/process.py")):
        importlib.import_module(f"lexicons.{process_file.parent.name}.process")

    return LexiconProcessor.__subclasses__()


@pytest.fixture(params=_discover_processors(), ids=lambda cls: cls.__name__)
def processor(request: pytest.FixtureRequest) -> LexiconProcessor:
    """
    Parameterizes processors for all tests
    """
    return request.param()


def test_base_dir_exists(processor: LexiconProcessor) -> None:
    """
    Checks that the lexicon's processing directory exists
    """
    assert processor.base_dir.exists()


def test_lexicon_name_matches_directory(processor: LexiconProcessor) -> None:
    """
    Checks that the name of the lexicon specified in the class is the same as the name of the
    lexicon directory in `lexicons`
    """
    assert processor.lexicon_name == processor.base_dir.name
