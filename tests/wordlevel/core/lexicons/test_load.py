import logging
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from wordlevel.core.lexicons.load import HF_HTTP_LOGGER, Lexicon, load_lexicon


CENTER = 5.0


@pytest.fixture
def lexicon_f(tmp_path: Path) -> Path:
    """
    Writes a lexicon parquet the way `LexiconProcessor.write` does, with the scores in the frame and
    the center score in the parquet's key-value metadata
    """
    lexicon_f = tmp_path / "lexicon.parquet"
    pl.DataFrame({"token": ["good", "bad"], "score": [8.0, 2.0]}).write_parquet(
        lexicon_f, metadata={"center": str(CENTER)}
    )

    return lexicon_f


@pytest.fixture
def lexicon(lexicon_f: Path, monkeypatch: pytest.MonkeyPatch) -> Lexicon:
    """
    Stubs out the HuggingFace download so `load_lexicon` reads the local parquet instead of calling
    out to the hub
    """
    monkeypatch.setattr(
        "wordlevel.core.lexicons.load.hf_hub_download", lambda **kwargs: str(lexicon_f)
    )

    return load_lexicon("labMT", "lexicon.parquet")


def test_center_read_from_parquet_metadata(lexicon: Lexicon) -> None:
    """
    Checks that the center score is read out of the parquet's key-value metadata and cast to a
    float, rather than being left as the string it is stored as
    """
    assert lexicon.center == CENTER


def test_scores_read_from_parquet(lexicon: Lexicon) -> None:
    """
    Checks that the lexicon's scores are scanned from the same parquet the center comes from
    """
    expected = pl.DataFrame({"token": ["good", "bad"], "score": [8.0, 2.0]})

    assert_frame_equal(lexicon.df, expected)


def test_hf_token_passed_to_download(lexicon_f: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Checks that a HuggingFace token is handed to the hub download, so that the request is
    authenticated rather than anonymous
    """
    download_kwargs: dict[str, object] = {}

    def fake_hf_hub_download(**kwargs: object) -> str:
        download_kwargs.update(kwargs)
        return str(lexicon_f)

    monkeypatch.setattr("wordlevel.core.lexicons.load.hf_hub_download", fake_hf_hub_download)

    load_lexicon("labMT", "lexicon.parquet", hf_token="hf_abc123")

    assert download_kwargs["token"] == "hf_abc123"


def test_hf_token_advisory_suppressed(
    lexicon_f: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Checks that HuggingFace's advisory about sending unauthenticated requests is filtered out of the
    logs while a lexicon is downloading, while its other messages still reach the user
    """
    hf_logger = logging.getLogger(HF_HTTP_LOGGER)

    def fake_hf_hub_download(**kwargs: object) -> str:
        hf_logger.warning(
            "Warning: You are sending unauthenticated requests to the HF Hub. Please set a "
            "HF_TOKEN to enable higher rate limits and faster downloads."
        )
        hf_logger.warning("Retrying in 1s [Retry 1/5].")
        return str(lexicon_f)

    monkeypatch.setattr("wordlevel.core.lexicons.load.hf_hub_download", fake_hf_hub_download)

    with caplog.at_level(logging.WARNING, logger=HF_HTTP_LOGGER):
        load_lexicon("labMT", "lexicon.parquet")

    assert caplog.messages == ["Retrying in 1s [Retry 1/5]."]


def test_hf_token_advisory_filter_removed(lexicon: Lexicon) -> None:
    """
    Checks that the filter is taken back off of HuggingFace's logger once the lexicon is loaded, so
    that it cannot silence messages logged outside of a download
    """
    assert logging.getLogger(HF_HTTP_LOGGER).filters == []
