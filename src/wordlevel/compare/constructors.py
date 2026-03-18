from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import polars as pl
from polars import col

from wordlevel.types import PolarsFrame
from wordlevel.compare.types import ComparatorData
from wordlevel.compare.constants import FREQ_PREFIX, NORMALIZATION_SUFFIX, LABEL
from wordlevel.expressions.utils import normalize_freqs


if TYPE_CHECKING:
    from wordlevel.compare.comparator import CorpusComparator


ListOfFreqs = Sequence[Mapping[str, float]]
DictOfFreqs = Mapping[str, Mapping[str, float]]


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
    fill_nulls: bool = True,
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

    # Do necessary transformations
    lf = df.clone().lazy()

    if fill_nulls:
        lf = lf.with_columns(*[col(freq_col).fill_null(0) for freq_col in freq_cols])

    if normalize:
        lf = _normalize_frequency_columns(
            lf,
            freq_cols=deduped_freq_cols,
            normalization_suffix=normalization_suffix,
        )
    else:
        normalization_suffix = ""

    return ComparatorData(
        df=lf,
        label_col=label_col,
        freq_cols=deduped_freq_cols,
        normalization_suffix=normalization_suffix,
        comparisons=dict(),
    )


def from_list_of_freqs(
    freqs: ListOfFreqs,
    corpora: Sequence[str] | None = None,
    label: str = LABEL,
    freq_prefix: str = FREQ_PREFIX,
    normalize: bool = True,
    normalization_suffix: str = NORMALIZATION_SUFFIX,
) -> ComparatorData:
    """ """
    if corpora is None:
        corpora = [f"corpus{n}" for n in range(1, len(freqs) + 1)]
    else:
        n_freqs = len(freqs)
        n_corpora = len(corpora)
        if n_freqs != n_corpora:
            raise ValueError(
                f"`freqs` and `corpora` must be the same length ({n_freqs} vs. {n_corpora})"
            )

    if len(freqs) < 2:
        raise ValueError("At least two frequency sets (corpora) need to be provided")

    return _from_freqs(
        freqs=zip(corpora, freqs),
        label=label,
        freq_prefix=freq_prefix,
        normalize=normalize,
        normalization_suffix=normalization_suffix,
    )


def from_dict_of_freqs(
    freqs: DictOfFreqs,
    label: str = LABEL,
    freq_prefix: str = FREQ_PREFIX,
    normalize: bool = True,
    normalization_suffix: str = NORMALIZATION_SUFFIX,
) -> ComparatorData:
    """ """
    if len(freqs) < 2:
        raise ValueError("At least two frequency sets (corpora) need to be provided")

    return _from_freqs(
        freqs=freqs.items(),
        label=label,
        freq_prefix=freq_prefix,
        normalize=normalize,
        normalization_suffix=normalization_suffix,
    )


def _from_freqs(
    freqs: Iterable[tuple[str, Mapping[str, float]]],
    label: str = LABEL,
    normalize: bool = True,
    freq_prefix: str = FREQ_PREFIX,
    normalization_suffix: str = NORMALIZATION_SUFFIX,
) -> ComparatorData:
    """ """
    freq_cols: list[str] = []
    lf = pl.LazyFrame(schema={label: pl.String})

    for corpus, word2freq in freqs:
        if not word2freq:
            raise ValueError(f"There are no word frequencies for `{corpus}`")

        freq_col = f"{freq_prefix}{corpus}"
        freq_cols.append(freq_col)

        words, word_freqs = zip(*word2freq.items())
        lf_corpus = pl.LazyFrame([words, word_freqs], schema=[label, freq_col])
        lf = lf.join(lf_corpus, on=label, how="outer")

    return from_polars(
        lf,
        label_col=label,
        freq_cols=freq_cols,
        fill_nulls=True,
        normalize=normalize,
        normalization_suffix=normalization_suffix,
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
