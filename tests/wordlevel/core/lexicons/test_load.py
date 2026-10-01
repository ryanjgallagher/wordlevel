import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import polars as pl
import pytest
from huggingface_hub._local_folder import write_download_metadata
from huggingface_hub.errors import (
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RevisionNotFoundError,
    XetDownloadError,
)
from huggingface_hub.file_download import HfFileMetadata
from polars.testing import assert_frame_equal

from wordlevel.core.lexicons.load import (
    HF_DOWNLOAD_LOGGER,
    HF_HTTP_LOGGER,
    Lexicon,
    LexiconConnectionError,
    LexiconDownloadError,
    load_lexicon,
    resolve_revision,
    stored_lexicon_paths,
    stored_revision,
    update_lexicons,
)


CENTER = 5.0
COMMIT = "a" * 40
# Attached to the stubbed error responses, which `httpx` requires; it is never sent
HF_REQUEST = httpx.Request("GET", "https://huggingface.co")


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

    return load_lexicon("labMT", "lexicon.parquet", revision=COMMIT)


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

    load_lexicon("labMT", "lexicon.parquet", revision=COMMIT, hf_token="hf_abc123")

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
    monkeypatch.setattr("wordlevel.core.lexicons.load.stored_revision", lambda **kwargs: COMMIT)

    load_lexicon("labMT", "lexicon.parquet", revision=COMMIT, cache_dir=tmp_path)

    assert download_kwargs["local_dir"] == tmp_path


def test_revision_passed_to_download(lexicon_f: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Checks that the lexicon is downloaded at the requested commit, rather than at whatever the
    repository's default branch points to
    """
    download_kwargs: dict[str, object] = {}

    def fake_hf_hub_download(**kwargs: object) -> str:
        download_kwargs.update(kwargs)
        return str(lexicon_f)

    monkeypatch.setattr("wordlevel.core.lexicons.load.hf_hub_download", fake_hf_hub_download)

    load_lexicon("labMT", "lexicon.parquet", revision=COMMIT)

    assert download_kwargs["revision"] == COMMIT


def test_revision_recorded_on_lexicon(lexicon: Lexicon) -> None:
    """
    Checks that the commit a lexicon was loaded at is recorded on it, so that the user can pin it
    """
    assert lexicon.revision == COMMIT


def test_stored_copy_at_other_commit_raised_as_connection_error(
    lexicon_f: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that when HuggingFace falls back to a stored copy at a different commit than the one
    requested, which it does silently when the hub cannot be reached, the load is raised rather
    than returning the wrong version of the lexicon
    """
    monkeypatch.setattr(
        "wordlevel.core.lexicons.load.hf_hub_download", lambda **kwargs: str(lexicon_f)
    )
    monkeypatch.setattr("wordlevel.core.lexicons.load.stored_revision", lambda **kwargs: "b" * 40)

    with pytest.raises(LexiconConnectionError):
        load_lexicon("labMT", "lexicon.parquet", revision=COMMIT, cache_dir=tmp_path)


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
        load_lexicon("labMT", "lexicon.parquet", revision=COMMIT)

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
        load_lexicon("labMT", "lexicon.parquet", revision=COMMIT)

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
        HfHubHTTPError("429 Too Many Requests", response=httpx.Response(429, request=HF_REQUEST)),
        RevisionNotFoundError(
            "404 Revision Not Found", response=httpx.Response(404, request=HF_REQUEST)
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
        load_lexicon("labMT", "lexicon.parquet", revision=COMMIT)

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
        load_lexicon("labMT", "lexicon.parquet", revision=COMMIT)


@pytest.mark.parametrize(
    "download_error",
    [
        LocalEntryNotFoundError("Could not connect to the hub"),
        OfflineModeIsEnabled("Offline mode is enabled"),
        HfHubHTTPError("429 Too Many Requests", response=httpx.Response(429, request=HF_REQUEST)),
        HfHubHTTPError("503 Service Unavailable", response=httpx.Response(503, request=HF_REQUEST)),
        httpx.ConnectError("Connection refused"),
        XetDownloadError("The transfer failed"),
    ],
    indirect=True,
)
def test_unreachable_hub_raised_as_connection_error(download_error: Exception) -> None:
    """
    Checks that failures to reach HuggingFace are raised as a `LexiconConnectionError`, so that
    callers can fall back to a stored lexicon when the hub cannot be reached
    """
    with pytest.raises(LexiconConnectionError):
        load_lexicon("labMT", "lexicon.parquet", revision=COMMIT)


@pytest.mark.parametrize(
    "download_error",
    [
        HfHubHTTPError("401 Unauthorized", response=httpx.Response(401, request=HF_REQUEST)),
        RevisionNotFoundError(
            "404 Revision Not Found", response=httpx.Response(404, request=HF_REQUEST)
        ),
    ],
    indirect=True,
)
def test_refused_request_not_raised_as_connection_error(download_error: Exception) -> None:
    """
    Checks that HuggingFace refusing a request is not raised as a `LexiconConnectionError`, so that
    callers do not fall back to a stored lexicon when the request itself is wrong
    """
    with pytest.raises(LexiconDownloadError) as error:
        load_lexicon("labMT", "lexicon.parquet", revision=COMMIT)

    assert not isinstance(error.value, LexiconConnectionError)


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
        load_lexicon("labMT", "lexicon.parquet", revision=COMMIT)

    assert logging.getLogger(logger_name).filters == []


def test_stored_revision_read_from_download_metadata(lexicon_f: Path, tmp_path: Path) -> None:
    """
    Checks that the commit a lexicon was downloaded at is read from the metadata that HuggingFace
    stores alongside it
    """
    cache_dir = tmp_path / "cache"
    repo_path = "labMT/data/processed/lexicon.parquet"
    (cache_dir / repo_path).parent.mkdir(parents=True)
    lexicon_f.rename(cache_dir / repo_path)
    write_download_metadata(cache_dir, repo_path, commit_hash=COMMIT, etag="etag")

    assert stored_revision("labMT", "lexicon.parquet", None, cache_dir) == COMMIT


def test_stored_revision_none_when_not_stored(tmp_path: Path) -> None:
    """
    Checks that a lexicon that has never been downloaded has no stored revision, rather than
    raising
    """
    assert stored_revision("labMT", "lexicon.parquet", None, tmp_path) is None


def test_stored_revision_none_without_download_metadata(lexicon_f: Path, tmp_path: Path) -> None:
    """
    Checks that a lexicon file that HuggingFace did not record a download for has no stored
    revision, so that it is downloaded again rather than trusted at an unknown commit
    """
    cache_dir = tmp_path / "cache"
    repo_path = "labMT/data/processed/lexicon.parquet"
    (cache_dir / repo_path).parent.mkdir(parents=True)
    lexicon_f.rename(cache_dir / repo_path)

    assert stored_revision("labMT", "lexicon.parquet", None, cache_dir) is None


def test_resolve_revision_requests_given_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Checks that the revision being resolved is the one requested from HuggingFace, and that the
    commit it reports is returned
    """
    url_kwargs: dict[str, object] = {}

    def fake_hf_hub_url(**kwargs: object) -> str:
        url_kwargs.update(kwargs)
        return "https://huggingface.co"

    monkeypatch.setattr("wordlevel.core.lexicons.load.hf_hub_url", fake_hf_hub_url)
    monkeypatch.setattr(
        "wordlevel.core.lexicons.load.get_hf_file_metadata",
        lambda url, token: HfFileMetadata(
            commit_hash=COMMIT, etag="etag", location=url, size=1, xet_file_data=None
        ),
    )

    commit = resolve_revision("labMT", "lexicon.parquet", None, "v1.0", hf_token=None)

    assert url_kwargs["revision"] == "v1.0"
    assert commit == COMMIT


def test_resolve_revision_raises_without_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Checks that a response that does not report a commit is raised, rather than returned as a
    revision that cannot be pinned
    """
    monkeypatch.setattr(
        "wordlevel.core.lexicons.load.get_hf_file_metadata",
        lambda url, token: HfFileMetadata(
            commit_hash=None, etag="etag", location=url, size=1, xet_file_data=None
        ),
    )

    with pytest.raises(LexiconDownloadError, match="did not report which commit"):
        resolve_revision("labMT", "lexicon.parquet", None, "main", hf_token=None)


def test_resolve_revision_unreachable_hub_raised_as_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Checks that failing to reach HuggingFace while resolving a revision is raised as a
    `LexiconConnectionError`, so that callers can fall back to a stored lexicon
    """

    def fake_get_hf_file_metadata(url: str, token: str | None) -> HfFileMetadata:
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(
        "wordlevel.core.lexicons.load.get_hf_file_metadata", fake_get_hf_file_metadata
    )

    with pytest.raises(LexiconConnectionError):
        resolve_revision("labMT", "lexicon.parquet", None, "main", hf_token=None)


def test_stored_lexicon_paths_skip_hf_metadata(tmp_path: Path) -> None:
    """
    Checks that stored lexicon files are listed by their repository paths, however deeply they are
    nested, while the files HuggingFace keeps in its `.cache` directory are left out
    """
    for repo_path in [
        "labMT/data/processed/english.parquet",
        "SocialSent/data/processed/1990.parquet",
        ".cache/huggingface/download/labMT/data/processed/english.parquet",
    ]:
        (tmp_path / repo_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / repo_path).touch()

    assert stored_lexicon_paths(tmp_path) == [
        "SocialSent/data/processed/1990.parquet",
        "labMT/data/processed/english.parquet",
    ]


class FakeHfApi:
    """
    Stands in for `HfApi`, reporting that the repository's default branch points to `COMMIT`
    """

    def repo_info(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(sha=COMMIT)


def test_update_lexicons_downloads_stored_paths_at_latest_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that only the given lexicon files are downloaded, and at the commit that the default
    branch was resolved to, so that every file is updated to the same version
    """
    snapshot_kwargs: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        snapshot_kwargs.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("wordlevel.core.lexicons.load.HfApi", FakeHfApi)
    monkeypatch.setattr("wordlevel.core.lexicons.load.snapshot_download", fake_snapshot_download)

    repo_paths = ["labMT/data/processed/english.parquet"]
    update_lexicons(cache_dir=tmp_path, repo_paths=repo_paths, hf_token=None)

    assert snapshot_kwargs["revision"] == COMMIT
    assert snapshot_kwargs["allow_patterns"] == repo_paths


def test_update_lexicons_unreachable_hub_raised_as_connection_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks that failing to reach HuggingFace while updating is raised, rather than the stored files
    being left as they are without the user knowing
    """

    class UnreachableHfApi:
        def repo_info(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
            raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr("wordlevel.core.lexicons.load.HfApi", UnreachableHfApi)

    with pytest.raises(LexiconConnectionError):
        update_lexicons(
            cache_dir=tmp_path,
            repo_paths=["labMT/data/processed/english.parquet"],
            hf_token=None,
        )
