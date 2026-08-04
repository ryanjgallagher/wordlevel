import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cached_property
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download, is_offline_mode
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    HTTPError,  # type: ignore[reportPrivateImportUsage]
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    XetDownloadError,
)


# The logger that `huggingface_hub` emits the hub's response header warnings on
HF_HTTP_LOGGER = "huggingface_hub.utils._http"


class LexiconDownloadError(Exception):
    """
    Raised when a lexicon could not be downloaded from HuggingFace
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
    """

    lf: pl.LazyFrame
    center: float
    token_col: str = "token"
    score_col: str = "score"

    @cached_property
    def df(self) -> pl.DataFrame:
        """
        The lexicon's tokens and their scores, collected into a dataframe
        """
        return self.lf.collect()


@contextmanager
def _suppress_hf_token_warning() -> Iterator[None]:
    """
    Silences the HuggingFace advisory that unauthenticated requests are being sent to the hub. The
    hub returns the advisory as an `X-HF-Warning` response header, which `huggingface_hub` emits as
    a log record rather than a warning, so it has to be filtered at the logging layer.
    """
    logger = logging.getLogger(HF_HTTP_LOGGER)

    def drop_token_advisory(record: logging.LogRecord) -> bool:
        return "HF_TOKEN" not in record.getMessage()

    logger.addFilter(drop_token_advisory)
    try:
        yield
    finally:
        logger.removeFilter(drop_token_advisory)


def load_lexicon(
    lexicon: str,
    filename: str,
    subfolder: str | None = None,
    cache_dir: str | Path | None = None,
    hf_token: str | None = None,
) -> Lexicon:
    """
    Downloads a lexicon from the HuggingFace dataset repository where the processed lexicons are
    published, reading the scores from the file's rows and the center of the scoring scale from its
    metadata

    Downloaded files are cached, so a lexicon is only retrieved over the network the first time it
    is loaded.

    Args:
        lexicon:
            The name of the lexicon
        filename:
            The name of the file holding the lexicon's scores within the lexicon's directory
        subfolder:
            A directory nested within the lexicon's directory that `filename` sits in
        cache_dir:
            The directory that downloaded lexicons are cached in
        hf_token:
            A HuggingFace access token to authenticate the download with. If not given, the token
            is read from the `HF_TOKEN` environment variable or a local HuggingFace login, if
            either is available.

    Returns:
        A `Lexicon` of tokens and their scores

    Raises:
        LexiconDownloadError: If the lexicon could not be downloaded from HuggingFace
    """
    full_subfolder = f"{lexicon}/data/processed"
    if subfolder is not None:
        full_subfolder = f"{full_subfolder}/{subfolder}"

    with _suppress_hf_token_warning():
        try:
            lexicon_f = hf_hub_download(
                repo_id="ryanjgallag/wordlevel-lexicons",
                subfolder=full_subfolder,
                filename=filename,
                repo_type="dataset",
                cache_dir=cache_dir,
                token=hf_token,
            )
        except (LocalEntryNotFoundError, OfflineModeIsEnabled) as error:
            # Offline mode is reported as a failure to find the file locally, so the two are only
            # told apart by asking HuggingFace whether offline mode is on
            message: str
            if is_offline_mode():
                message = (
                    f"Could not download the '{lexicon}' lexicon because HuggingFace's offline "
                    "mode is enabled. Unset the 'HF_HUB_OFFLINE' environment variable to download "
                    "it."
                )
            else:
                message = (
                    f"Could not reach HuggingFace to download the '{lexicon}' lexicon. Check your "
                    "network connection and try again."
                )

            raise LexiconDownloadError(message) from error
        except EntryNotFoundError as error:
            raise LexiconDownloadError(
                f"The file '{full_subfolder}/{filename}' is not published on HuggingFace"
            ) from error
        except (GatedRepoError, RepositoryNotFoundError) as error:
            raise LexiconDownloadError(
                f"Could not access the HuggingFace repository holding the '{lexicon}' lexicon. If "
                "you are using a HuggingFace token, check that it is valid."
            ) from error
        except HfHubHTTPError as error:
            raise LexiconDownloadError(
                f"HuggingFace returned an error while downloading the '{lexicon}' lexicon: {error}."
                " If you are being rate limited, pass a HuggingFace token to load the lexicon with."
            ) from error
        except (XetDownloadError, HTTPError) as error:
            raise LexiconDownloadError(
                f"The download of the '{lexicon}' lexicon from HuggingFace failed: {error}. Check "
                "your network connection and try again."
            ) from error

    lf = pl.scan_parquet(lexicon_f)
    center_score = float(pl.read_parquet_metadata(lexicon_f)["center"])

    return Lexicon(lf=lf, center=center_score)
