from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from functools import cached_property
from types import MappingProxyType

import polars as pl

from wordlevel.api.catalog.constructors import (
    from_catalog,
    from_polars,
    from_dict_of_freqs,
    from_list_of_freqs,
)
from wordlevel.api.catalog.plot import Plotter
from wordlevel.api.catalog.constructors import CatalogData, DictOfFreqs, ListOfFreqs
from wordlevel.api.comparison.comparison import Comparison, ComparisonResult


class Catalog:
    """
    The workspace for corpora and making comparisons between them.

    For specific construction functions, see the [construction](construct/index.md) documentation.

    Args:
        data:
            The corpora token-frequency data. Accepts a Polars DataFrame or LazyFrame,
            [`ListOfFreqs`][wordlevel.api.catalog.constructors.ListOfFreqs],
            [`DictOfFreqs`][wordlevel.api.catalog.constructors.DictOfFreqs], or an existing
            `Catalog`. The class will attempt to infer the structure of the data and use the
            appropriate constructor.
        token_col:
            If the `data` is a Polars dataframe, the column that contains the tokens (words,
            subwords, n-grams, etc.). Ignored otherwise
        corpora:
            If the `data` is a [`ListOfFreqs`][wordlevel.api.catalog.constructors.ListOfFreqs], the
            names of the corpora, listed in the same order as `data`. If not provided, will default
            to `corpus1`, `corpus2`, etc. If the `data` is a Polars Dataframe or Lazyframe, the
            names of the columns containing token frequencies, which must be provided. Ignored
            otherwise

    Raises:
        TypeError: If `data` is not one of the supported types.
    """

    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame | Catalog | CatalogData | DictOfFreqs | ListOfFreqs,
        token_col: str | None = None,
        corpora: Sequence[str] | None = None,
    ) -> None:
        if isinstance(data, CatalogData):
            init_data = data
        elif isinstance(data, Catalog):
            init_data = from_catalog(data)
        elif isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            if token_col is None:
                raise ValueError("`token_col` must be specified if providing a DataFrame")
            if corpora is None:
                raise ValueError("`corpora` must be specified if providing a DataFrame")
            init_data = from_polars(
                df=data,
                token_col=token_col,
                freq_cols=corpora,
            )
        elif isinstance(data, Mapping):
            init_data = from_dict_of_freqs(freqs=data)
        elif isinstance(data, Sequence):
            init_data = from_list_of_freqs(freqs=data, corpora=corpora)
        else:
            raise TypeError(f"Input type not supported: {type(data)}")

        self._init_from_catalog_data(init_data)

    def _init_from_catalog_data(self, data: CatalogData) -> None:
        """
        Initializes the catalog given the core `CatalogData`.
        """
        self._df = data.df
        self._token_col = data.token_col
        self._freq_cols = set(data.freq_cols)
        self._normalization_suffix = data.normalization_suffix
        self._comparisons = data.comparisons

    @classmethod
    def from_catalog(cls, catalog: Catalog) -> Catalog:
        """
        Creates a new catalog from an existing one.

        The new catalog copies the data of the original one without mutating it.

        Example:
            ```python
            --8<-- "snippets/catalog/catalog.py:from_catalog"
            ```

        Args:
            catalog: An existing `Catalog`

        Returns:
            A new `Catalog` with the same internal data as the provided catalog
        """
        init_data = from_catalog(catalog)

        return cls(data=init_data)

    @classmethod
    def from_polars(
        cls,
        df: pl.DataFrame | pl.LazyFrame,
        token_col: str,
        corpora: Collection[str],
        normalize: bool = True,
    ) -> Catalog:
        """
        Creates a `Catalog` of corpora from a Polars dataframe.

        Example:
            ```python
            --8<-- "snippets/catalog/catalog.py:from_polars"
            ```

        Args:
            df:
                A Polars DataFrame or LazyFrame containing corpus information. Rows are tokens and
                columns are their frequencies in different corpora
            token_col:
                The column that contains the tokens (words, subwords, n-grams, etc.)
            corpora:
                The columns that include the token frequencies for each corpus
            normalize:
                Whether to normalize the corpus frequencies for each corpus so that they are in the
                range $[0, 1]$ and sum to 1. The original corpus frequencies will be preserved

        Returns:
            A `Catalog` of corpora representing the provided tokens and frequencies

        Raises:
            KeyError: If `token_col` is not a column in `df`.
            KeyError: If any column in `corpora` is not a column in `df`.
            ValueError: If fewer than two columns are provided in `corpora`.
        """
        init_data = from_polars(
            df=df,
            token_col=token_col,
            freq_cols=corpora,
            normalize=normalize,
        )

        return cls(data=init_data)

    @classmethod
    def from_list_of_freqs(
        cls,
        freqs: ListOfFreqs,
        corpora: Sequence[str] | None = None,
    ) -> Catalog:
        """
        Creates a `Catalog` of corpora from a sequence of token-to-frequency mappings, each
        representing the frequencies of tokens in a particular corpus.

        Example:
            ```python
            --8<-- "snippets/catalog/catalog.py:from_list_of_freqs"
            ```

        Args:
            freqs:
                A sequence of token-to-frequency mappings, one per corpus. Each mapping is
                `{token: frequency}` for a single corpus
            corpora:
                The names of the corpora, listed in the same order as `freqs`. If not provided, will
                default to `corpus1`, `corpus2`, etc.

        Returns:
            A `Catalog` of corpora representing the provided tokens and frequencies

        Raises:
            ValueError: If `freqs` and `corpora` have different lengths.
            ValueError: If fewer than two corpora are provided in `freqs`.
            ValueError: If any corpus in `freqs` has no token frequencies.
        """
        init_data = from_list_of_freqs(freqs=freqs, corpora=corpora)

        return cls(data=init_data)

    @classmethod
    def from_dict_of_freqs(cls, freqs: DictOfFreqs) -> Catalog:
        """
        Creates a `Catalog` of corpora from named token-to-frequency mappings.

        Example:
            ```python
            --8<-- "snippets/catalog/catalog.py:from_dict_of_freqs"
            ```

        Args:
            freqs:
                A mapping of corpus names to token-to-frequency mappings, one per corpus. Each
                mapping is `{corpus: {token: frequency}}`, with one mapping per corpus

        Returns:
            A `Catalog` of corpora representing the provided tokens and frequencies

        Raises:
            ValueError: If fewer than two corpora are provided in `freqs`.
            ValueError: If any corpus in `freqs` has no token frequencies.
        """
        init_data = from_dict_of_freqs(freqs=freqs)

        return cls(data=init_data)

    @property
    def comparisons(self) -> Mapping[str, ComparisonResult]:
        """
        Comparisons made between corpora in the catalog, keyed by each comparison's alias.
        """
        return MappingProxyType(self._comparisons)

    @property
    def df(self) -> pl.DataFrame:
        """
        Dataframe representing the frequencies of tokens in the corpora in the catalog.
        """
        return self._df.collect()

    @cached_property
    def plot(self) -> Plotter:
        """
        Namespace for plotting methods. See [available plots](plot/index.md).
        """
        return Plotter(self)

    def with_comparisons(self, *comparisons: Comparison) -> Catalog:
        """
        Returns a new catalog with comparisons added to it.

        New comparisons will overwrite existing comparisons with the same name.

        Example:
            ```python
            --8<-- "snippets/catalog/catalog.py:setup"

            --8<-- "snippets/catalog/catalog.py:with_comparisons"
            ```

        Args:
            comparisons: The comparisons to add to the catalog

        Returns:
            The catalog with the comparisons added to it
        """
        catalog = self.__class__(self)

        for comparison in comparisons:
            result = ComparisonResult.from_comparison(
                comparison,
                self._df,
                token_col=self._token_col,
                normalization_suffix=self._normalization_suffix,
            )
            catalog._comparisons[result.alias] = result

        return catalog
