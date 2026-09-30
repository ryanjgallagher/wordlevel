import logging
from pathlib import Path

import httpx
import polars as pl
import pytest
from huggingface_hub.errors import (
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    XetDownloadError,
)
from polars.testing import assert_frame_equal

from wordlevel.core.lexicons.load import (
    HF_DOWNLOAD_LOGGER,
    HF_HTTP_LOGGER,
    Lexicon,
    LexiconDownloadError,
    load_lexicon,
)


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


def test_cache_dir_passed_as_local_dir(
    lexicon_f: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that lexicons are stored in a plain directory rather than HuggingFace's versioned cache,
    so that each lexicon keeps the commit it was downloaded at, independently of the others
    """
    download_kwargs: dict[str, object] = {}

    def fake_hf_hub_download(**kwargs: object) -> str:
        download_kwargs.update(kwargs)
        return str(lexicon_f)

    monkeypatch.setattr("wordlevel.core.lexicons.load.hf_hub_download", fake_hf_hub_download)

    load_lexicon("labMT", "lexicon.parquet", cache_dir=tmp_path)

    assert download_kwargs["local_dir"] == tmp_path


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


def test_hub_fallback_message_suppressed(
    lexicon_f: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Checks that HuggingFace's message about using a stored file without checking the hub is
    filtered out of the logs, while its other messages still reach the user
    """
    hf_logger = logging.getLogger(HF_DOWNLOAD_LOGGER)

    def fake_hf_hub_download(**kwargs: object) -> str:
        hf_logger.warning(
            "Couldn't access the Hub to check for update but local file already exists. "
            "Defaulting to existing file. (error: Cannot access file since 'local_files_only=True' "
            "as been set.)"
        )
        hf_logger.warning("Some other message.")
        return str(lexicon_f)

    monkeypatch.setattr("wordlevel.core.lexicons.load.hf_hub_download", fake_hf_hub_download)

    with caplog.at_level(logging.WARNING, logger=HF_DOWNLOAD_LOGGER):
        load_lexicon("labMT", "lexicon.parquet")

    assert caplog.messages == ["Some other message."]


@pytest.mark.parametrize("logger_name", [HF_HTTP_LOGGER, HF_DOWNLOAD_LOGGER])
def test_hf_log_filters_removed(lexicon: Lexicon, logger_name: str) -> None:
    """
    Checks that the filters are taken back off of HuggingFace's loggers once the lexicon is loaded,
    so that they cannot silence messages logged outside of a download
    """
    assert logging.getLogger(logger_name).filters == []


@pytest.fixture
def download_error(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Exception:
    """
    Stubs the HuggingFace download so that it fails with the error being parameterized over
    """
    download_error: Exception = request.param

    def fake_hf_hub_download(**kwargs: object) -> str:
        raise download_error

    monkeypatch.setattr("wordlevel.core.lexicons.load.hf_hub_download", fake_hf_hub_download)

    return download_error


@pytest.mark.parametrize(
    "download_error",
    [
        LocalEntryNotFoundError("Could not connect to the hub"),
        OfflineModeIsEnabled("Offline mode is enabled"),
        HfHubHTTPError(
            "429 Too Many Requests",
            response=httpx.Response(429, request=httpx.Request("GET", "https://huggingface.co")),
        ),
        XetDownloadError("The transfer failed"),
    ],
    indirect=True,
)
def test_download_failures_raised_as_download_error(download_error: Exception) -> None:
    """
    Checks that HuggingFace's failures are surfaced as a `LexiconDownloadError`, so that the user
    gets an interpretable message rather than a `huggingface_hub` traceback, and that the original
    error is kept as the cause so that it is still available for debugging
    """
    with pytest.raises(LexiconDownloadError) as error:
        load_lexicon("labMT", "lexicon.parquet")

    assert error.value.__cause__ is download_error


@pytest.mark.parametrize(
    "download_error", [LocalEntryNotFoundError("Could not connect to the hub")], indirect=True
)
def test_offline_mode_reported_instead_of_network_failure(
    download_error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that a user who has turned HuggingFace's offline mode on is told so, rather than being
    told to check a network connection that was never used. HuggingFace reports offline mode as a
    failure to find the file locally, so the two are only told apart by asking it whether offline
    mode is on
    """
    monkeypatch.setattr("wordlevel.core.lexicons.load.is_offline_mode", lambda: True)

    with pytest.raises(LexiconDownloadError, match="offline mode is enabled"):
        load_lexicon("labMT", "lexicon.parquet")


@pytest.mark.parametrize(
    "download_error", [LocalEntryNotFoundError("Could not connect to the hub")], indirect=True
)
@pytest.mark.parametrize("logger_name", [HF_HTTP_LOGGER, HF_DOWNLOAD_LOGGER])
def test_hf_log_filters_removed_on_error(download_error: Exception, logger_name: str) -> None:
    """
    Checks that the filters are taken back off of HuggingFace's loggers when the download fails,
    and not just when it succeeds
    """
    with pytest.raises(LexiconDownloadError):
        load_lexicon("labMT", "lexicon.parquet")

    assert logging.getLogger(logger_name).filters == []
