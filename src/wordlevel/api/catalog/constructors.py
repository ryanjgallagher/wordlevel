from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence, TypeAlias

import polars as pl
from polars import col

from wordlevel.constants import NORMALIZATION_SUFFIX, TOKEN
from wordlevel.core.utils import normalize_freqs


if TYPE_CHECKING:
    from wordlevel.api.catalog.catalog import Catalog
    from wordlevel.api.comparison.comparison import ComparisonResult


@dataclass
class CatalogData:
    df: pl.LazyFrame
    token_col: str
    freq_cols: Collection[str]
    normalization_suffix: str
    comparisons: dict[str, ComparisonResult]


ListOfFreqs: TypeAlias = Sequence[Mapping[str, float]]
"""
A sequence of word-to-frequencey mappings, one per corpus. Each mapping is `{word: frequency}` for a single corpus

Example:
    ```python
    list_of_freqs = [
        {"hi": 1, "hey": 2, "hello": 3},
        {"hi": 2, "hey"; 4, "bye": 6}
    ]
    ```
"""


DictOfFreqs: TypeAlias = Mapping[str, Mapping[str, float]]
"""
A mapping of corpus names to word-to-frequency mappings. The mapping is `{corpus: {word: frequency}}`

Example:
    ```python
    dict_of_freqs = {
        "corpus1": {"hi": 1, "hey": 2, "hello": 3},
        "corpus2": {"hi": 2, "hey"; 4, "bye": 6}
    }
    ```
"""


def from_catalog(catalog: Catalog) -> CatalogData:
    """ """
    return CatalogData(
        df=catalog._df.clone().lazy(),
        token_col=catalog._token_col,
        freq_cols=catalog._freq_cols,
        normalization_suffix=catalog._normalization_suffix,
        comparisons=catalog._comparisons.copy(),
    )


def from_polars(
    df: pl.DataFrame | pl.LazyFrame,
    token_col: str,
    freq_cols: Collection[str],
    normalize: bool = True,
) -> CatalogData:
    """ """
    # Check all columns exist
    deduped_freq_cols = set(freq_cols)

    if isinstance(df, pl.DataFrame):
        cols = set(df.columns)
    else:
        cols = set(df.collect_schema().names())

    if token_col not in cols:
        raise KeyError(f"Token column not in provided dataframe: {token_col}")

    if len(deduped_freq_cols) < 2:
        raise ValueError("At least two frequency columns must be provided via `freq_cols`")

    for freq_col in deduped_freq_cols:
        if freq_col not in cols:
            raise KeyError(f"Frequency column not in provided dataframe: {freq_col}")

    # Do necessary transformations
    lf = df.clone().lazy()
    lf = lf.with_columns(*[col(freq_col).fill_null(0) for freq_col in deduped_freq_cols])

    if normalize:
        lf = _normalize_frequency_columns(
            lf,
            freq_cols=deduped_freq_cols,
            normalization_suffix=NORMALIZATION_SUFFIX,
        )
        normalization_suffix = NORMALIZATION_SUFFIX
    else:
        normalization_suffix = ""

    return CatalogData(
        df=lf,
        token_col=token_col,
        freq_cols=deduped_freq_cols,
        normalization_suffix=normalization_suffix,
        comparisons=dict(),
    )


def from_list_of_freqs(
    freqs: ListOfFreqs,
    corpora: Sequence[str] | None = None,
) -> CatalogData:
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
        token=TOKEN,
    )


def from_dict_of_freqs(
    freqs: DictOfFreqs,
) -> CatalogData:
    """ """
    if len(freqs) < 2:
        raise ValueError("At least two frequency sets (corpora) need to be provided")

    return _from_freqs(
        freqs=freqs.items(),
        token=TOKEN,
    )


def _from_freqs(
    freqs: Iterable[tuple[str, Mapping[str, float]]],
    token: str = TOKEN,
) -> CatalogData:
    """ """
    freq_cols: list[str] = []
    lf = pl.LazyFrame(schema={token: pl.String})

    for corpus, token2freq in freqs:
        if not token2freq:
            raise ValueError(f"There are no token frequencies for `{corpus}`")

        freq_cols.append(corpus)

        tokens, token_freqs = zip(*token2freq.items())
        lf_corpus = pl.LazyFrame([tokens, token_freqs], schema=[token, corpus])
        lf = lf.join(lf_corpus, on=token, how="full", coalesce=True)

    return from_polars(
        lf,
        token_col=token,
        freq_cols=freq_cols,
    )


def _normalize_frequency_columns(
    df: pl.LazyFrame,
    freq_cols: Collection[str],
    normalization_suffix: str = NORMALIZATION_SUFFIX,
) -> pl.LazyFrame:
    """ """
    normed_cols = [
        normalize_freqs(col(freq_col)).alias(f"{freq_col}{normalization_suffix}")
        for freq_col in freq_cols
    ]

    return df.with_columns(*normed_cols)
