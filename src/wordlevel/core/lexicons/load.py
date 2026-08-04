from functools import cached_property
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download


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
    """
    full_subfolder = f"{lexicon}/data/processed"
    if subfolder is not None:
        full_subfolder = f"{full_subfolder}/{subfolder}"

    lexicon_f = hf_hub_download(
        repo_id="ryanjgallag/wordlevel-lexicons",
        subfolder=full_subfolder,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
        token=hf_token,
    )

    lf = pl.scan_parquet(lexicon_f)
    center_score = float(pl.read_parquet_metadata(lexicon_f)["center"])

    return Lexicon(lf=lf, center=center_score)
