import pytest

from lexicons import huggingface


def test_download_lexicon_fails_when_the_repository_has_no_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Checks that a lexicon with no data in the repository fails, rather than downloading nothing and
    reporting success
    """
    monkeypatch.setattr(huggingface, "list_lexicon_files", lambda lexicon, data_stage: [])
    monkeypatch.setattr(
        huggingface, "snapshot_download", lambda **kwargs: pytest.fail("A download was attempted")
    )

    with pytest.raises(FileNotFoundError):
        huggingface.download_lexicon("labMT", "raw")
