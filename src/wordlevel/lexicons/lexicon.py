from dataclasses import dataclass

from pathlib import Path
import polars as pl

from platformdirs import user_cache_dir
from huggingface_hub import hf_hub_download

from wordlevel.lexicons.types.labmt import labMTLanguage
from wordlevel.lexicons.types.socialsent import SocialSentYear


@dataclass
class Lexicon:
    lf: pl.LazyFrame
    center: float
    label_col: str = "token"
    score_col: str = "score"


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


class LexiconLoader:
    def __init__(self, cache_dir: str | None = None):
        """ """
        if cache_dir is None:
            cache_dir = user_cache_dir("wordlevel/lexicons")

        self.cache_dir = cache_dir

    def labMT(self, language: labMTLanguage) -> Lexicon:
        """ """
        return load_lexicon(
            lexicon="labMT", filename=f"{language}.parquet", cache_dir=self.cache_dir
        )

    def SocialSent(self, year: SocialSentYear) -> Lexicon:
        """ """
        return load_lexicon(
            lexicon="SocialSent", filename=f"{year}.parquet", cache_dir=self.cache_dir
        )


lex = LexiconLoader()
