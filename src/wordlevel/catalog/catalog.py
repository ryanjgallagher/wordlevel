from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from functools import cached_property
from types import MappingProxyType

import polars as pl

from wordlevel.catalog.constructors import (
    from_catalog,
    from_polars,
    from_dict_of_freqs,
    from_list_of_freqs,
)
from wordlevel.catalog.plot import Plotter
from wordlevel.catalog.types import CatalogData, DictOfFreqs, ListOfFreqs
from wordlevel.comparison.comparison import Comparison, ComparisonResult


class Catalog:
    """
    This is the docstring of the catalog

    Args:
        data:
            description
    """

    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame | Catalog | CatalogData | DictOfFreqs | ListOfFreqs,
        token_col: str | None = None,
        freq_cols: Collection[str] | None = None,
        corpora: Sequence[str] | None = None,
    ) -> None:
        """ """
        if isinstance(data, CatalogData):
            init_data = data
        elif isinstance(data, Catalog):
            init_data = from_catalog(data)
        elif isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            if token_col is None:
                raise ValueError("`token_col` must be specified if providing a DataFrame")
            if freq_cols is None:
                raise ValueError("`freq_cols` must be specified if providing a DataFrame")
            init_data = from_polars(
                df=data,
                token_col=token_col,
                freq_cols=freq_cols,
            )
        elif isinstance(data, Mapping):
            init_data = from_dict_of_freqs(freqs=data)
        elif isinstance(data, Sequence):
            init_data = from_list_of_freqs(freqs=data, corpora=corpora)
        else:
            raise TypeError(f"Input type not supported: {type(data)}")

        self._init_from_catalog_data(init_data)

    def _init_from_catalog_data(self, data: CatalogData) -> None:
        """ """
        self._df = data.df
        self._token_col = data.token_col
        self._freq_cols = set(data.freq_cols)
        self._normalization_suffix = data.normalization_suffix
        self._comparisons = data.comparisons

    @classmethod
    def from_catalog(cls, catalog: Catalog) -> Catalog:
        """
        Creates a `Catalog` from an existing `Catalog`. It is relatively cheap to
        copy the data from one catalog to the other.

        Args:
            catalog:
                An existing, initialized `Catalog`

        Returns:
            A new `Catalog` with the same internal data as the input
        """
        init_data = from_catalog(catalog)

        return cls(data=init_data)

    @classmethod
    def from_polars(
        cls,
        df: pl.DataFrame | pl.LazyFrame,
        token_col: str,
        freq_cols: Collection[str],
        normalize: bool = True,
    ) -> Catalog:
        """
        Creates a `Catalog` from a Polars dataframe.

        Args:
            df:
                A Polars data/lazyframe containing corpus information. Rows are tokens and columns
                are their frequencies in different corpora
            token_col:
                The column that contains the tokens (words, subwords, n-grams, etc.)
            freq_cols:
                The column or columns that include the frequencies for each corpus
            normalize:
                Whether to normalize the corpus frequencies for each corpus. The original corpus
                frequencies will be preserved
        """
        init_data = from_polars(
            df=df,
            token_col=token_col,
            freq_cols=freq_cols,
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
        Creates a `Catalog` from a list of frequencies, a sequence of word-to-frequency
        mappings.

        Args:
            freqs:
                A sequence of word-to-frequencey mappings, one per corpus. Each mapping is `{word: frequency}` for a single corpus
            corpora:
                The names of the corpora, listed in the same order as `freqs`. If not provided, will default to `corpus1`, `corpus2`, etc.

        Returns:
            A Catalog instance
        """
        init_data = from_list_of_freqs(freqs=freqs, corpora=corpora)

        return cls(data=init_data)

    @classmethod
    def from_dict_of_freqs(cls, freqs: DictOfFreqs) -> Catalog:
        """ """
        init_data = from_dict_of_freqs(freqs=freqs)

        return cls(data=init_data)

    @property
    def comparisons(self) -> Mapping[str, ComparisonResult]:
        """
        This is a description
        """
        return MappingProxyType(self._comparisons)

    @property
    def df(self) -> pl.DataFrame:
        """
        TODO: Change variable name of internal variable to `_lf`. Collect and update similar to
              a ComparisonResult
        """
        return self._df.collect()

    @cached_property
    def plot(self) -> Plotter:
        """ """
        return Plotter(self)

    def with_comparisons(self, *comparisons: Comparison) -> Catalog:
        """
        TODO: use pl.collect_all
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
