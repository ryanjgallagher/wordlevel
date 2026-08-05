from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import narwhals as nw
import polars as pl
from narwhals.typing import IntoFrame
from polars import col

from wordlevel.constants import NORMALIZATION_SUFFIX, TOKEN
from wordlevel.core.utils import normalize_freqs
from wordlevel.datasets.dataset import Dataset


if TYPE_CHECKING:
    from wordlevel.api.catalog.catalog import Catalog
    from wordlevel.api.comparison.comparison import ComparisonResult


@dataclass
class CatalogData:
    """
    The core data that is required to initialize and copy a `Catalog`

    Args:
        df: Polars
            DataFrame representing the frequencies of tokens in the corpora in the catalog.
        token_col:
            The column in `df` that contains the tokens (words, subwords, n-grams, etc.)
        freq_cols:
            The columns in `df` that include the token frequencies for each corpus
        normalization_suffix:
            The suffix used on the `freq_cols` to indicate their normalized forms where all
            frequencies are in the range $[0, 1]$ and sum to 1
        comparisons:
            Comparisons made between corpora in the catalog, keyed by each comparison's alias
    """

    df: pl.LazyFrame
    token_col: str
    freq_cols: Collection[str]
    normalization_suffix: str
    comparisons: dict[str, ComparisonResult]


ListOfFreqs: TypeAlias = Sequence[Mapping[str, float]]
"""
A sequence of token-to-frequency mappings, one per corpus. Each mapping is `{word: frequency}` for 
a single corpus

Example:
    ```python
    list_of_freqs = [
        {"hi": 1, "hey": 2, "hello": 3},
        {"hi": 2, "hey": 4, "bye": 6}
    ]
    ```
"""


DictOfFreqs: TypeAlias = Mapping[str, Mapping[str, float]]
"""
A mapping of corpus names to word-to-frequency mappings. The mapping is 
`{corpus: {word: frequency}}`

Example:
    ```python
    dict_of_freqs = {
        "corpus1": {"hi": 1, "hey": 2, "hello": 3},
        "corpus2": {"hi": 2, "hey": 4, "bye": 6}
    }
    ```
"""


def is_dataframe(data: object) -> bool:
    """
    Checks whether an object is a dataframe that can be converted to a Polars dataframe.

    Recognizes any dataframe from a backend supported by Narwhals, as well as any object that
    produces an Arrow C stream.

    Args:
        data: Any object

    Returns:
        Whether the object is a convertible dataframe
    """
    if isinstance(nw.from_native(data, pass_through=True), (nw.DataFrame, nw.LazyFrame)):
        return True

    # Producers of an Arrow C stream that Narwhals has no dedicated backend for
    return hasattr(data, "__arrow_c_stream__")


def from_catalog(catalog: Catalog) -> CatalogData:
    """
    Creates the data for a new catalog from an existing one.

    Copies the data of the original catalog without mutating it.

    Args:
        catalog: An existing `Catalog`

    Returns:
        The `CatalogData` needed to initialize a new catalog
    """
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
    """
    Creates the data for a new catalog from a Polars dataframe.

    Args:
        df:
            A Polars DataFrame or LazyFrame containing corpus information. Rows are tokens and
            columns are their frequencies in different corpora
        token_col:
            The column that contains the tokens (words, subwords, n-grams, etc.)
        freq_cols:
            The columns that include the token frequencies for each corpus
        normalize:
            Whether to normalize the corpus frequencies for each corpus so that they are in the
            range $[0, 1]$ and sum to 1. The original corpus frequencies will be preserved

    Returns:
        The `CatalogData` needed to initialize a new catalog

    Raises:
        KeyError: If `token_col` is not a column in `df`.
        KeyError: If any column in `freq_cols` is not a column in `df`.
        ValueError: If fewer than two columns are provided in `freq_cols`.
    """
    # Check all columns exist
    deduped_freq_cols = set(freq_cols)

    if isinstance(df, pl.DataFrame):
        cols = set(df.columns)
    else:
        cols = set(df.collect_schema().names())

    if token_col not in cols:
        raise KeyError(f"Token column not in provided dataframe: {token_col}")

    for freq_col in deduped_freq_cols:
        if freq_col not in cols:
            raise KeyError(f"Frequency column not in provided dataframe: {freq_col}")

    if len(deduped_freq_cols) < 2:
        raise ValueError("At least two frequency columns must be provided via `freq_cols`")

    # Do necessary transformations
    lf = df.clone().lazy()
    lf = lf.select(token_col, *[col(freq_col).fill_null(0) for freq_col in deduped_freq_cols])

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
        comparisons={},
    )


def from_dataframe(
    df: IntoFrame,
    token_col: str,
    freq_cols: Collection[str],
    normalize: bool = True,
) -> CatalogData:
    """
    Creates the data for a new catalog from any dataframe supported by Narwhals.

    The dataframe is converted to Polars before the catalog is constructed. Polars dataframes are
    used as given, so a LazyFrame stays lazy. Dataframes from other lazy backends, like DuckDB and
    PySpark, are materialized in memory.

    Args:
        df:
            A dataframe containing corpus information, from any backend supported by Narwhals.
            Rows are tokens and columns are their frequencies in different corpora
        token_col:
            The column that contains the tokens (words, subwords, n-grams, etc.)
        freq_cols:
            The columns that include the token frequencies for each corpus
        normalize:
            Whether to normalize the corpus frequencies for each corpus so that they are in the
            range $[0, 1]$ and sum to 1. The original corpus frequencies will be preserved

    Returns:
        The `CatalogData` needed to initialize a new catalog

    Raises:
        TypeError: If `df` is not a dataframe that can be converted to Polars.
        KeyError: If `token_col` is not a column in `df`.
        KeyError: If any column in `freq_cols` is not a column in `df`.
        ValueError: If fewer than two columns are provided in `freq_cols`.
    """
    df_pl: pl.DataFrame | pl.LazyFrame
    df_nw = nw.from_native(df, pass_through=True)

    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        df_pl = df
    elif isinstance(df_nw, (nw.DataFrame, nw.LazyFrame)):
        # Narwhals has no lazy-to-lazy bridge into Polars, so lazy backends are materialized
        df_pl = df_nw.lazy().collect(backend="polars").to_polars()
    elif hasattr(df, "__arrow_c_stream__"):
        # Producers of an Arrow C stream that Narwhals also supports
        df_pl = nw.from_arrow(df, backend=pl).to_polars()
    else:
        raise TypeError(f"Input type not supported: {type(df)}")

    return from_polars(
        df=df_pl,
        token_col=token_col,
        freq_cols=freq_cols,
        normalize=normalize,
    )


def from_wl_dataset(data: Dataset, corpora: Collection[str] | None = None) -> CatalogData:
    """
    Creates the data for a new catalog from a wl.Dataset

    Args:
        data:
            A WordLevel dataset
        corpora:
            The names of the corpora to include

    Returns:
        The `CatalogData` needed to initialize a new catalog

    Raises:
        KeyError: If any corpus in `corpora` is not a valid specifier
        ValueError: If fewer than two `corpora` are provided
    """
    df = data.as_df()

    if corpora is None:
        freq_cols = df.select(pl.all().exclude(data._token_col)).columns
    else:
        freq_cols = corpora

    return from_polars(df=df, token_col=data._token_col, freq_cols=freq_cols)


def from_list_of_freqs(
    freqs: ListOfFreqs,
    corpora: Sequence[str] | None = None,
) -> CatalogData:
    """
    Creates the data for a new catalog from a sequence of token-to-frequency mappings, each
    representing the frequencies of tokens in a particular corpus.

    Args:
        freqs:
            A sequence of token-to-frequency mappings, one per corpus. Each mapping is
            `{token: frequency}` for a single corpus
        corpora:
            The names of the corpora, listed in the same order as `freqs`. If not provided, will
            default to `corpus1`, `corpus2`, etc.

    Returns:
        The `CatalogData` needed to initialize a new catalog

    Raises:
        ValueError: If `freqs` and `corpora` have different lengths.
        ValueError: If fewer than two corpora are provided in `freqs`.
        ValueError: If any corpus in `freqs` has no token frequencies.
    """
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
    """
    Creates the data for a new catalog from named token-to-frequency mappings.

    Args:
        freqs:
            A mapping of corpus names to token-to-frequency mappings, one per corpus. Each
            mapping is `{corpus: {token: frequency}}`, with one mapping per corpus

    Returns:
        The `CatalogData` needed to initialize a new catalog

    Raises:
        ValueError: If fewer than two corpora are provided in `freqs`.
        ValueError: If any corpus in `freqs` has no token frequencies.
    """
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
    """
    Creates the data for a new catalog from token-frequency data structures

    Args:
        freqs:
            A sequence of tuples, one per corpus. Each tuple is `(corpus, {token: frequency})`
        token:
            The name the use for the column representing tokens in the produced catalog dataframe

    Returns:
        The `CatalogData` needed to initialize a new catalog
    """
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
    """
    Normalizes columns in a dataframe so all values are in the range $[0, 1]$ and sum to 1. The
    normalized columns are named using a `normalization_suffix`, appended to the original column
    names.
    """
    normed_cols = [
        normalize_freqs(col(freq_col)).alias(f"{freq_col}{normalization_suffix}")
        for freq_col in freq_cols
    ]

    return df.with_columns(*normed_cols)
