import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import polars as pl
from huggingface_hub import (
    DryRunFileInfo,
    HfApi,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
    is_offline_mode,
    snapshot_download,
)
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    HTTPError,  # type: ignore[reportPrivateImportUsage]
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    XetDownloadError,
)


# The HuggingFace dataset repository that the processed lexicons are published to
LEXICON_REPO_ID = "ryanjgallag/wordlevel-lexicons"

# The hub's warning that unauthenticated requests are being sent
HF_HTTP_LOGGER = "huggingface_hub.utils._http"
HF_TOKEN_ADVISORY_TEXT = "HF_TOKEN"
# `huggingface_hub`'s warning that a stored file was used without checking the hub for an update
HF_DOWNLOAD_LOGGER = "huggingface_hub.file_download"
HF_STORED_FALLBACK_TEXT = "Couldn't access the Hub to check for update"


class LexiconDownloadError(Exception):
    """
    Raised when a lexicon could not be downloaded from HuggingFace
    """


class LexiconConnectionError(LexiconDownloadError):
    """
    Raised when HuggingFace could not be reached to retrieve a lexicon, as opposed to HuggingFace
    refusing the request
    """


@dataclass(frozen=True)
class Lexicon:
    """
    A set of scores assigned to tokens, along with the center of the scoring scale

    Attributes:
        lf:
            The (lazy) dataframe containing the lexicon's tokens and their scores. Contains one row
            per token.
        center:
            The center of the lexicon's scoring scale, which scores are interpreted relative to
        token_col:
            The column in `lf` that contains the tokens
        score_col:
            The column in `lf` that contains the scores
        revision:
            The commit of the HuggingFace repository that the lexicon was downloaded at, if it was
            downloaded from HuggingFace
    """

    lf: pl.LazyFrame
    center: float
    token_col: str = "token"
    score_col: str = "score"
    revision: str | None = None

    @cached_property
    def df(self) -> pl.DataFrame:
        """
        The lexicon's tokens and their scores, collected into a dataframe
        """
        return self.lf.collect()


@contextmanager
def _suppress_hf_log(logger_name: str, log_match_text: str) -> Generator[None, None, None]:
    """
    Silences the records that `huggingface_hub` logs containing `log_match_text`.
    `huggingface_hub` reports some of its messages as log records rather than warnings, so they
    have to be filtered at the logging layer.

    Args:
        logger_name:
            The name of the `huggingface_hub` logger that the records are emitted on
        log_match_text:
            The text that identifies the records to silence
    """
    logger = logging.getLogger(logger_name)

    def drop_record(record: logging.LogRecord) -> bool:
        return log_match_text not in record.getMessage()

    logger.addFilter(drop_record)
    try:
        yield
    finally:
        logger.removeFilter(drop_record)


@contextmanager
def _translate_hf_errors(
    subject: str, repo_path: str, revision: str
) -> Generator[None, None, None]:
    """
    Re-raises the errors that are raised when a connection to HuggingFace fails as a
    `LexiconDownloadError`, so that the user gets an interpretable message. Failures to reach
    HuggingFace at all are raised as a `LexiconConnectionError`.

    Args:
        subject:
            What is being retrieved, phrased to complete "Could not download ...", for the error
            messages (e.g. "the 'labMT' lexicon")
        repo_path:
            The path of the file being retrieved within the HuggingFace repository, for the error
            messages
        revision:
            The revision of the HuggingFace repository being retrieved, for the error messages
    """
    try:
        yield
    except (LocalEntryNotFoundError, OfflineModeIsEnabled) as error:
        # Offline mode is reported as a failure to find the file locally, so the two are only told
        # apart by asking HuggingFace whether offline mode is on
        message: str
        if is_offline_mode():
            message = (
                f"Could not download {subject} because HuggingFace's offline mode is enabled. "
                "Unset the 'HF_HUB_OFFLINE' environment variable to allow downloads."
            )
        else:
            message = (
                f"Could not reach HuggingFace to download {subject}. Check your network "
                "connection and try again."
            )

        raise LexiconConnectionError(message) from error
    except EntryNotFoundError as error:
        raise LexiconDownloadError(
            f"The file '{repo_path}' is not published on HuggingFace"
        ) from error
    except (GatedRepoError, RepositoryNotFoundError) as error:
        raise LexiconDownloadError(
            f"Could not access the HuggingFace repository holding {subject}. If you are using a "
            "HuggingFace token, check that it is valid."
        ) from error
    except RevisionNotFoundError as error:
        raise LexiconDownloadError(
            f"The revision '{revision}' does not exist in the HuggingFace repository holding "
            f"{subject}"
        ) from error
    except HfHubHTTPError as error:
        # Rate limiting and server errors mean the hub could not serve the request
        status_code = error.response.status_code
        error_type: type[LexiconDownloadError]
        if status_code == 429 or status_code >= 500:
            error_type = LexiconConnectionError
        else:
            error_type = LexiconDownloadError

        raise error_type(
            f"HuggingFace returned an error while downloading {subject}: {error}. If you are "
            "being rate limited, pass a HuggingFace token to authenticate the download with."
        ) from error
    except (XetDownloadError, HTTPError) as error:
        raise LexiconConnectionError(
            f"The download of {subject} from HuggingFace failed: {error}. Check your network "
            "connection and try again."
        ) from error


def _repo_subfolder(lexicon: str, subfolder: str | None) -> str:
    """
    The directory within the HuggingFace repository that holds a lexicon's processed files

    Args:
        lexicon:
            The name of the lexicon
        subfolder:
            A directory nested within the lexicon's directory that its file sits in
    """
    full_subfolder = f"{lexicon}/data/processed"
    if subfolder is not None:
        full_subfolder = f"{full_subfolder}/{subfolder}"

    return full_subfolder


def stored_revision(
    lexicon: str, filename: str, subfolder: str | None, cache_dir: str | Path
) -> str | None:
    """
    Reads the commit of the HuggingFace repository that a stored lexicon was downloaded at, without
    calling out to the hub

    Args:
        lexicon:
            The name of the lexicon
        filename:
            The name of the file holding the lexicon's scores within the lexicon's directory
        subfolder:
            A directory nested within the lexicon's directory that `filename` sits in
        cache_dir:
            The directory that downloaded lexicons are stored in

    Returns:
        The commit hash of the stored lexicon, or `None` if the lexicon is not stored
    """
    with _suppress_hf_log(HF_DOWNLOAD_LOGGER, HF_STORED_FALLBACK_TEXT):
        try:
            download_info = hf_hub_download(
                repo_id=LEXICON_REPO_ID,
                subfolder=_repo_subfolder(lexicon, subfolder),
                filename=filename,
                repo_type="dataset",
                local_dir=cache_dir,
                local_files_only=True,
                dry_run=True,
            )
        except LocalEntryNotFoundError:
            return None

    # A file stored without HuggingFace's download metadata is returned as a plain path, and has
    # no commit to report
    if not isinstance(download_info, DryRunFileInfo):
        return None

    return download_info.commit_hash


def resolve_revision(
    lexicon: str, filename: str, subfolder: str | None, revision: str, hf_token: str | None
) -> str:
    """
    Asks HuggingFace which commit a revision of the lexicon repository currently points to

    Args:
        lexicon:
            The name of the lexicon
        filename:
            The name of the file holding the lexicon's scores within the lexicon's directory
        subfolder:
            A directory nested within the lexicon's directory that `filename` sits in
        revision:
            A branch, tag, or commit hash of the lexicon repository
        hf_token:
            A HuggingFace access token to authenticate the request with. If not given, the token is
            read from the `HF_TOKEN` environment variable or a local HuggingFace login, if either
            is available.

    Returns:
        The commit hash that `revision` points to

    Raises:
        LexiconConnectionError: If HuggingFace could not be reached
        LexiconDownloadError: If HuggingFace refused the request, or did not report a commit
    """
    repo_path = f"{_repo_subfolder(lexicon, subfolder)}/{filename}"

    with (
        _suppress_hf_log(HF_HTTP_LOGGER, HF_TOKEN_ADVISORY_TEXT),
        _translate_hf_errors(
            subject=f"the '{lexicon}' lexicon", repo_path=repo_path, revision=revision
        ),
    ):
        file_metadata = get_hf_file_metadata(
            hf_hub_url(
                repo_id=LEXICON_REPO_ID,
                filename=repo_path,
                repo_type="dataset",
                revision=revision,
            ),
            token=hf_token,
        )

    if file_metadata.commit_hash is None:
        raise LexiconDownloadError(
            f"HuggingFace did not report which commit revision '{revision}' of the '{lexicon}' "
            "lexicon points to. If you are behind a proxy, check that it passes HuggingFace's "
            "response headers through."
        )

    return file_metadata.commit_hash


def load_lexicon(
    lexicon: str,
    filename: str,
    revision: str,
    subfolder: str | None = None,
    cache_dir: str | Path | None = None,
    hf_token: str | None = None,
) -> Lexicon:
    """
    Downloads a lexicon from the HuggingFace dataset repository where the processed lexicons are
    published, reading the scores from the file's rows and the center of the scoring scale from its
    metadata

    A lexicon already stored at `revision` is read without calling out to the hub.

    Args:
        lexicon:
            The name of the lexicon
        filename:
            The name of the file holding the lexicon's scores within the lexicon's directory
        revision:
            The commit hash of the lexicon repository to load the lexicon at
        subfolder:
            A directory nested within the lexicon's directory that `filename` sits in
        cache_dir:
            The directory that downloaded lexicons are stored in, mirroring their paths within the
            HuggingFace repository
        hf_token:
            A HuggingFace access token to authenticate the download with. If not given, the token
            is read from the `HF_TOKEN` environment variable or a local HuggingFace login, if
            either is available.

    Returns:
        A `Lexicon` of tokens and their scores

    Raises:
        LexiconConnectionError: If HuggingFace could not be reached to download the lexicon
        LexiconDownloadError: If the lexicon could not be downloaded from HuggingFace
    """
    full_subfolder = _repo_subfolder(lexicon, subfolder)

    with (
        _suppress_hf_log(HF_HTTP_LOGGER, HF_TOKEN_ADVISORY_TEXT),
        _suppress_hf_log(HF_DOWNLOAD_LOGGER, HF_STORED_FALLBACK_TEXT),
        _translate_hf_errors(
            subject=f"the '{lexicon}' lexicon",
            repo_path=f"{full_subfolder}/{filename}",
            revision=revision,
        ),
    ):
        lexicon_f = hf_hub_download(
            repo_id=LEXICON_REPO_ID,
            subfolder=full_subfolder,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=cache_dir,
            token=hf_token,
        )

    # When the hub cannot be reached, HuggingFace returns the stored file whatever commit it was
    # downloaded at, so check that it is the one requested
    if cache_dir is not None:
        downloaded_commit = stored_revision(
            lexicon=lexicon, filename=filename, subfolder=subfolder, cache_dir=cache_dir
        )
        if downloaded_commit != revision:
            raise LexiconConnectionError(
                f"Could not reach HuggingFace to download revision '{revision}' of the "
                f"'{lexicon}' lexicon. Check your network connection and try again."
            )

    lf = pl.scan_parquet(lexicon_f)
    center_score = float(pl.read_parquet_metadata(lexicon_f)["center"])

    return Lexicon(lf=lf, center=center_score, revision=revision)


def stored_lexicon_paths(cache_dir: str | Path) -> list[str]:
    """
    Lists the lexicon files stored in a directory, by their paths within the HuggingFace repository

    Args:
        cache_dir:
            The directory that downloaded lexicons are stored in

    Returns:
        The repository paths of the stored lexicon files
    """
    cache_path = Path(cache_dir)

    return sorted(
        lexicon_f.relative_to(cache_path).as_posix()
        for lexicon_f in cache_path.rglob("*.parquet")
        # HuggingFace keeps its download metadata in a `.cache` directory alongside the files
        if ".cache" not in lexicon_f.relative_to(cache_path).parts
    )


def update_lexicons(cache_dir: str | Path, repo_paths: list[str], hf_token: str | None) -> None:
    """
    Downloads the latest commit of the lexicon repository's default branch for each of the given
    stored lexicon files, only transferring the files whose contents have changed.

    Args:
        cache_dir:
            The directory that downloaded lexicons are stored in
        repo_paths:
            The repository paths of the lexicon files to update
        hf_token:
            A HuggingFace access token to authenticate the requests with. If not given, the token
            is read from the `HF_TOKEN` environment variable or a local HuggingFace login, if
            either is available.

    Raises:
        LexiconConnectionError: If HuggingFace could not be reached
        LexiconDownloadError: If HuggingFace refused the request, or did not report a commit
    """
    subject = "the cached lexicons"

    # Resolved up front because `snapshot_download` returns the stored files without raising when
    # the hub cannot be reached. Only published files are downloaded, so the repository stands in
    # for a single file's path.
    with (
        _suppress_hf_log(HF_HTTP_LOGGER, HF_TOKEN_ADVISORY_TEXT),
        _translate_hf_errors(subject=subject, repo_path=LEXICON_REPO_ID, revision="main"),
    ):
        latest_commit = HfApi().repo_info(LEXICON_REPO_ID, repo_type="dataset", token=hf_token).sha

    if latest_commit is None:
        raise LexiconDownloadError(
            "HuggingFace did not report which commit the lexicon repository's default branch "
            "points to. If you are behind a proxy, check that it passes HuggingFace's responses "
            "through."
        )

    with (
        _suppress_hf_log(HF_HTTP_LOGGER, HF_TOKEN_ADVISORY_TEXT),
        _translate_hf_errors(subject=subject, repo_path=LEXICON_REPO_ID, revision=latest_commit),
    ):
        snapshot_download(
            repo_id=LEXICON_REPO_ID,
            repo_type="dataset",
            revision=latest_commit,
            local_dir=cache_dir,
            allow_patterns=repo_paths,
            token=hf_token,
        )
