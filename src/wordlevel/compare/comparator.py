from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from functools import cached_property
from types import MappingProxyType

import polars as pl

from wordlevel.compare.constructors import (
    from_comparator,
    from_polars,
    from_dict_of_freqs,
    from_list_of_freqs,
)
from wordlevel.compare.plot import Plotter
from wordlevel.compare.types import ComparatorData, DictOfFreqs, ListOfFreqs
from wordlevel.compare.comparisons.comparison import Comparison, ComparisonResult


class CorpusComparator:
    """
    This is the docstring of the comparator

    Args:
        data:
            description
    """

    def __init__(
        self,
        data: pl.DataFrame
        | pl.LazyFrame
        | CorpusComparator
        | ComparatorData
        | DictOfFreqs
        | ListOfFreqs,
        label_col: str | None = None,
        freq_cols: Collection[str] | None = None,
        corpora: Sequence[str] | None = None,
    ) -> None:
        """ """
        if isinstance(data, ComparatorData):
            init_data = data
        elif isinstance(data, CorpusComparator):
            init_data = from_comparator(data)
        elif isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            if label_col is None:
                raise ValueError("`label_col` must be specified if providing a DataFrame")
            if freq_cols is None:
                raise ValueError("`freq_cols` must be specified if providing a DataFrame")
            init_data = from_polars(
                df=data,
                label_col=label_col,
                freq_cols=freq_cols,
            )
        elif isinstance(data, Mapping):
            init_data = from_dict_of_freqs(freqs=data)
        elif isinstance(data, Sequence):
            init_data = from_list_of_freqs(freqs=data, corpora=corpora)
        else:
            raise TypeError(f"Input type not supported: {type(data)}")

        self._init_from_comparator_data(init_data)

    def _init_from_comparator_data(self, data: ComparatorData) -> None:
        """ """
        self._df = data.df
        self._label_col = data.label_col
        self._freq_cols = set(data.freq_cols)
        self._normalization_suffix = data.normalization_suffix
        self._comparisons = data.comparisons

    @classmethod
    def from_comparator(cls, comparator: CorpusComparator) -> CorpusComparator:
        """
        Creates a `CorpusComparator` from an existing `CorpusComparator`. It is relatively cheap to
        copy the data from one comparator to the other.

        Args:
            comparator:
                An existing, initialized `CorpusComparator`

        Returns:
            A new `CorpusComparator` with the same internal data as the input
        """
        init_data = from_comparator(comparator)

        return cls(data=init_data)

    @classmethod
    def from_polars(
        cls,
        df: pl.DataFrame | pl.LazyFrame,
        label_col: str,
        freq_cols: Collection[str],
        normalize: bool = True,
    ) -> CorpusComparator:
        """
        Creates a `CorpusComparator` from a Polars dataframe.

        Args:
            df:
                A Polars data/lazyframe containing corpus information. Rows are words and columns
                are their frequencies in different corpora
            label_col:
                The column that includes the "labels" or "words"
            freq_cols:
                The column or columns that include the frequencies for each corpus
            normalize:
                Whether to normalize the corpus frequencies for each corpus. The original corpus
                frequencies will be preserved
        """
        init_data = from_polars(
            df=df,
            label_col=label_col,
            freq_cols=freq_cols,
            normalize=normalize,
        )

        return cls(data=init_data)

    @classmethod
    def from_list_of_freqs(
        cls,
        freqs: ListOfFreqs,
        corpora: Sequence[str] | None = None,
    ) -> CorpusComparator:
        """
        Creates a `CorpusComparator` from a list of frequencies, a sequence of word-to-frequency
        mappings.

        Args:
            freqs:
                A sequence of word-to-frequencey mappings, one per corpus. Each mapping is `{word: frequency}` for a single corpus
            corpora:
                The names of the corpora, listed in the same order as `freqs`. If not provided, will default to `corpus1`, `corpus2`, etc.

        Returns:
            A CorpusComparator instance
        """
        init_data = from_list_of_freqs(freqs=freqs, corpora=corpora)

        return cls(data=init_data)

    @classmethod
    def from_dict_of_freqs(cls, freqs: DictOfFreqs) -> CorpusComparator:
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

    def with_comparisons(self, *comparisons: Comparison) -> CorpusComparator:
        """
        TODO: use pl.collect_all
        """
        comparator = self.__class__(self)

        for comparison in comparisons:
            result = ComparisonResult.from_comparison(
                comparison,
                self._df,
                label_col=self._label_col,
                normalization_suffix=self._normalization_suffix,
            )
            comparator._comparisons[result.alias] = result

        return comparator
