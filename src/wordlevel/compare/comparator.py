from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from functools import cached_property
from types import MappingProxyType
from typing_extensions import Self

import polars as pl

from wordlevel.types import PolarsFrame
from wordlevel.compare.constructors import (
    from_comparator,
    from_polars,
    from_dict_of_freqs,
    from_list_of_freqs,
    DictOfFreqs,
    ListOfFreqs,
)
from wordlevel.compare.constants import FREQ_PREFIX, LABEL, NORMALIZATION_SUFFIX
from wordlevel.compare.plot import Plotter
from wordlevel.compare.types import ComparatorData
from wordlevel.compare.comparisons.comparison import Comparison, ComparisonResult


class CorpusComparator:
    """ """

    def __init__(
        self,
        data: PolarsFrame | CorpusComparator | ComparatorData | DictOfFreqs | ListOfFreqs,
        label_col: str | None = None,
        freq_cols: Collection[str] | None = None,
        corpora: Sequence[str] | None = None,
        label: str = LABEL,
        freq_prefix: str = FREQ_PREFIX,
        fill_nulls: bool = True,
        normalize: bool = True,
        normalization_suffix: str = NORMALIZATION_SUFFIX,
    ) -> None:
        """ """
        if isinstance(data, ComparatorData):
            init_data = data
        elif isinstance(data, CorpusComparator):
            init_data = from_comparator(data)
        elif isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            init_data = from_polars(
                df=data,
                label_col=label_col,
                freq_cols=freq_cols,
                fill_nulls=fill_nulls,
                normalize=normalize,
                normalization_suffix=normalization_suffix,
            )
        elif isinstance(data, Mapping):
            init_data = from_dict_of_freqs(
                freqs=data,
                label=label,
                freq_prefix=freq_prefix,
                normalize=normalize,
                normalization_suffix=normalization_suffix,
            )
        elif isinstance(data, Sequence):
            init_data = from_list_of_freqs(
                freqs=data,
                corpora=corpora,
                label=label,
                freq_prefix=freq_prefix,
                normalize=normalize,
                normalization_suffix=normalization_suffix,
            )
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
        """ """
        init_data = from_comparator(comparator)

        return cls(data=init_data)

    @classmethod
    def from_polars(
        cls,
        df: pl.DataFrame,
        label_col: str,
        freq_cols: Collection[str],
        fill_nulls: bool = True,
        normalize: bool = True,
        normalization_suffix: str = NORMALIZATION_SUFFIX,
    ) -> Self:
        """ """
        init_data = from_polars(
            df=df,
            label_col=label_col,
            freq_cols=freq_cols,
            fill_nulls=fill_nulls,
            normalize=normalize,
            normalization_suffix=normalization_suffix,
        )

        return cls(data=init_data)

    @classmethod
    def from_list_of_freqs(
        cls,
        freqs: ListOfFreqs,
        corpora: Sequence[str] | None = None,
        label: str = LABEL,
        freq_prefix: str = FREQ_PREFIX,
        normalize: bool = True,
        normalization_suffix: str = NORMALIZATION_SUFFIX,
    ) -> Self:
        """ """
        init_data = from_list_of_freqs(
            freqs=freqs,
            corpora=corpora,
            label=label,
            freq_prefix=freq_prefix,
            normalize=normalize,
            normalization_suffix=normalization_suffix,
        )

        return cls(data=init_data)

    @classmethod
    def from_dict_of_freqs(
        cls,
        freqs: DictOfFreqs,
        label: str = LABEL,
        freq_prefix: str = FREQ_PREFIX,
        normalize: bool = True,
        normalization_suffix: str = NORMALIZATION_SUFFIX,
    ) -> Self:
        """ """
        init_data = from_dict_of_freqs(
            freqs=freqs,
            label=label,
            freq_prefix=freq_prefix,
            normalize=normalize,
            normalization_suffix=normalization_suffix,
        )

        return cls(data=init_data)

    @property
    def comparisons(self) -> Mapping[str, ComparisonResult]:
        """"""
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
            comparison = comparison._with_normalization_suffix(self._normalization_suffix)
            result = ComparisonResult.from_comparison(comparison, self._df)
            comparator._comparisons[result.alias] = result

        return comparator
