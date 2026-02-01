from __future__ import annotations

from collections.abc import Collection
from types import MappingProxyType
from typing import Self, Mapping

import polars as pl

from wordlevel.types import PolarsFrame
from wordlevel.compare.constructors import from_comparator, from_polars
from wordlevel.compare.constants import NORMALIZATION_SUFFIX
from wordlevel.compare.types import ComparatorData
from wordlevel.compare.comparisons.comparison import Comparison


class CorpusComparator:
    """ """

    def __init__(
        self,
        data: PolarsFrame | CorpusComparator | ComparatorData,
        label_col: str | None = None,
        freq_cols: Collection[str] | None = None,
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
        normalize: bool = True,
        normalization_suffix: str = NORMALIZATION_SUFFIX,
    ) -> Self:
        """ """
        init_data = from_polars(
            df=df,
            label_col=label_col,
            freq_cols=freq_cols,
            normalize=normalize,
            normalization_suffix=normalization_suffix,
        )

        return cls(data=init_data)

    @property
    def comparisons(self) -> Mapping[str, Comparison]:
        """"""
        return MappingProxyType(self._comparisons)

    @property
    def df(self) -> pl.DataFrame:
        """ """
        return self._df.collect()

    # def with_comparisons(self, *comparisons: Comparison) -> CorpusComparator:
    #     """
    #     TODO: don't just store the raw comparison, need to apply the "plan" (function) to the df

    #     tc = (
    #     Comparator(data=data, subsets=['1990', '2000', '2010'])
    #         .with_comparsions(
    #             comp('1990', '2000').score.rank_divergence().filter(min_scoore=20).alias('my_rank_div'),
    #             comp('2000', '2010').score.jensen_shannon()
    #         )
    #         .plot.bars('my_rank_div')
    #     )
    #     """
    #     comparator = self.__class__(self)

    #     for comparison in comparisons:
    #         comparison._normalization_suffix = self._normalization_suffix
    #         comparator.comparisons[comparison._alias] = comparison

    #     return comparator
