import polars as pl
from polars import col

from wordlevel.constants import (
    FREQ_CMP_COL,
    FREQ_REF_COL,
    NORMALIZATION_SUFFIX,
    NORMED_FREQ_CMP_COL,
    NORMED_FREQ_REF_COL,
)
from wordlevel.core.utils import normalize_freqs


def select_corpora(
    df: pl.DataFrame | pl.LazyFrame,
    token_col: str,
    freqs_ref_col: str,
    freqs_cmp_col: str,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Narrows a catalog's data down to the two corpora being compared, renaming their frequency
    columns to fixed internal names.

    Every operation that follows refers to the two corpora by those names rather than by the corpus
    names the user chose
    """
    return df.select(
        col(token_col),
        col(freqs_ref_col).alias(FREQ_REF_COL),
        col(freqs_cmp_col).alias(FREQ_CMP_COL),
    )


def renormalize_freqs(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Adds columns representing the normalized frequencies of the two corpora being compared.
    """
    return df.with_columns(
        normalize_freqs(col(FREQ_REF_COL)).alias(NORMED_FREQ_REF_COL),
        normalize_freqs(col(FREQ_CMP_COL)).alias(NORMED_FREQ_CMP_COL),
    )


def restore_corpus_names(
    df: pl.DataFrame | pl.LazyFrame,
    reference: str,
    comparison: str,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Renames the internal frequency columns back to names derived from the two corpora.

    `strict=False` allows the rename to only affect the columns that a scoring method uses, since
    most only use either the raw or normalized frequency columns
    """
    return df.rename(
        {
            FREQ_REF_COL: reference,
            FREQ_CMP_COL: comparison,
            NORMED_FREQ_REF_COL: f"{reference}{NORMALIZATION_SUFFIX}",
            NORMED_FREQ_CMP_COL: f"{comparison}{NORMALIZATION_SUFFIX}",
        },
        strict=False,
    )
