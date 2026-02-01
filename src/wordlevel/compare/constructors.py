from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

import polars as pl
from polars import col

from wordlevel.types import PolarsFrame
from wordlevel.compare.types import ComparatorData
from wordlevel.compare.constants import NORMALIZATION_SUFFIX
from wordlevel.expressions.utils import normalize_freqs


if TYPE_CHECKING:
    from wordlevel.compare.comparator import CorpusComparator


def from_comparator(comparator: CorpusComparator) -> ComparatorData:
    """ """
    return ComparatorData(
        df=comparator._df.clone().lazy(),
        label_col=comparator._label_col,
        freq_cols=comparator._freq_cols,
        normalization_suffix=comparator._normalization_suffix,
        comparisons=comparator._comparisons.copy(),
    )


def from_polars(
    df: PolarsFrame,
    label_col: str,
    freq_cols: Collection[str],
    normalize: bool = True,
    normalization_suffix: str = NORMALIZATION_SUFFIX,
) -> ComparatorData:
    """ """
    # Check all parameters specified
    if label_col is None:
        raise ValueError("`label_col` must be specified if providing a DataFrame")
    if freq_cols is None:
        raise ValueError("`freq_cols` must be specified if providing a DataFrame")

    # Check all columns exist
    deduped_freq_cols = set(freq_cols)

    if isinstance(df, pl.DataFrame):
        cols = set(df.columns)
    elif isinstance(df, pl.LazyFrame):
        cols = set(df.collect_schema().names())

    if label_col not in cols:
        raise KeyError(f"Label column not in provided dataframe: {label_col}")

    if len(deduped_freq_cols) < 2:
        raise ValueError("At least two frequency columns must be provided via `freq_cols`")

    for freq_col in deduped_freq_cols:
        if freq_col not in cols:
            raise KeyError(f"Frequency column not in provided dataframe: {freq_col}")

    if normalize:
        df_normed = _normalize_frequency_columns(
            df.clone().lazy(),
            freq_cols=deduped_freq_cols,
            normalization_suffix=normalization_suffix,
        )
    else:
        df_normed = df.clone().lazy()
        normalization_suffix = ""

    return ComparatorData(
        df=df_normed,
        label_col=label_col,
        freq_cols=deduped_freq_cols,
        normalization_suffix=normalization_suffix,
        comparisons=dict(),
    )


def _normalize_frequency_columns(
    df: PolarsFrame, freq_cols: Collection[str], normalization_suffix: str = NORMALIZATION_SUFFIX
) -> PolarsFrame:
    """ """
    normed_cols = [
        normalize_freqs(col(freq_col)).alias(f"{freq_col}{normalization_suffix}")
        for freq_col in freq_cols
    ]

    return df.with_columns(*normed_cols)
