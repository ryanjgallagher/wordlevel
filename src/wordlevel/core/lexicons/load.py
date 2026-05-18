from functools import cached_property
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download


@dataclass
class Lexicon:
    lf: pl.LazyFrame
    center: float
    token_col: str = "token"
    score_col: str = "score"

    @cached_property
    def df(self) -> pl.DataFrame:
        return self.lf.collect()


def load_lexicon(
    lexicon: str,
    filename: str,
    subfolder: str | None = None,
    cache_dir: str | Path | None = None,
) -> Lexicon:
    """ """
    full_subfolder = f"{lexicon}/data/processed"
    if subfolder is not None:
        full_subfolder = f"{full_subfolder}/{subfolder}"

    lexicon_f = hf_hub_download(
        repo_id="ryanjgallag/wordlevel-lexicons",
        subfolder=full_subfolder,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    lf = pl.scan_parquet(lexicon_f)
    center_score = float(pl.read_parquet_metadata(lexicon_f)["center"])

    return Lexicon(lf=lf, center=center_score)
