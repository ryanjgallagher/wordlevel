import importlib
import pathlib

import pytest

import wordlevel.api.lexicons.types
from lexicons.generate_types import enumerate_processors
from lexicons.processor import LexiconProcessor


@pytest.fixture(params=enumerate_processors().values(), ids=lambda cls: cls.__name__)
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


def test_list_local_raw_files_returns_paths_relative_to_raw_dir(
    processor: LexiconProcessor, tmp_path: pathlib.Path
) -> None:
    """
    Checks that local raw data is listed relative to `data/raw`, matching how the files are named
    on HuggingFace, and that a directory holding no files does not read as raw data
    """
    (tmp_path / "english").mkdir()
    (tmp_path / "english" / "labMTwords-english.csv").touch()
    (tmp_path / "empty").mkdir()

    processor.raw_dir = tmp_path

    assert processor._list_local_raw_files() == ["english/labMTwords-english.csv"]


def test_resolve_raw_files_prefers_local_data(
    processor: LexiconProcessor, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that HuggingFace is not consulted when the lexicon's raw data is already local
    """
    (tmp_path / "1800.tsv").touch()
    processor.raw_dir = tmp_path

    monkeypatch.setattr(
        processor, "_list_remote_raw_files", lambda: pytest.fail("HuggingFace was consulted")
    )

    assert processor._resolve_raw_files() == ["1800.tsv"]


def test_process_fails_when_no_data_is_written(
    processor: LexiconProcessor, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that processing which produces no data fails, rather than leaving an empty directory
    behind and reporting success
    """
    processor.raw_dir = tmp_path / "raw"
    processor.processed_dir = tmp_path / "processed"

    monkeypatch.setattr(processor, "_process_lexicon", lambda: None)

    with pytest.raises(FileNotFoundError):
        processor.process()


def test_types_dir_matches_package_types_module(processor: LexiconProcessor) -> None:
    """
    Checks that the processor writes generated types into the same directory that `wordlevel`
    imports them from. The processor builds its path from `base_dir`, so this catches the two
    drifting apart if the package is restructured
    """
    package_types_dir = pathlib.Path(wordlevel.api.lexicons.types.__file__).parent

    assert processor.types_dir == package_types_dir


def test_types_file_is_importable(processor: LexiconProcessor) -> None:
    """
    Checks that the types file the processor writes is the module that `wordlevel` imports, which
    the directory alone does not guarantee if the file is named differently than the module
    """
    module = importlib.import_module(
        f"wordlevel.api.lexicons.types.{processor.lexicon_name.lower()}"
    )

    assert pathlib.Path(module.__file__) == processor.types_f
