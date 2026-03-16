from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import altair as alt
import polars as pl
from polars import col

from wordlevel.compare.comparisons.comparison import ComparisonResult
from wordlevel.constants import (
    ATTRIBUTION_COL,
    PLOT_CONTRIBUTION_COL,
    PLOT_CONTRIBUTION_TYPE_COL,
    RANK_COL,
    SCORE_COL,
    COMPARISON_ATTRIBUTION,
    REFERENCE_ATTRIBUTION,
)
from wordlevel.expressions.shift.plot import bar_plot_df, shift_plot_df
from wordlevel.plots.shift.bars import (
    RankAxisConfig,
    ScoreAxisConfig,
    InternalScoreAxisConfig,
    BarConfig,
    InternalBarConfig,
)
from wordlevel.plots.shift.labels import LabelConfig
from wordlevel.plots.shift.shift import shift

if TYPE_CHECKING:
    from wordlevel.compare.comparator import CorpusComparator


PlotType = Literal["bar", "shift"]


@dataclass(frozen=True)
class PlotCacheKey:
    alias: str
    plot_type: PlotType
    split_nonnegative_by_corpus: bool


@dataclass
class PlotCachedDataFrame:
    df: pl.DataFrame
    max_rank: int


class Plotter:
    """ """

    def __init__(self, comparator: CorpusComparator):
        """ """
        self._comparator = comparator
        self._plot_dfs: dict[PlotCacheKey, PlotCachedDataFrame] = dict()

    def bar(
        self,
        comparison: str,
        max_rank: int = 50,
        vertical: bool = True,
        height: float | None = None,
        width: float | None = None,
        autoscale_axes: bool = True,
        rank_axis: RankAxisConfig | dict | None = None,
        score_axis: ScoreAxisConfig | dict | None = None,
        bar_config: BarConfig | dict | None = None,
        label_config: LabelConfig | dict | None = None,
        scale_factor: float = 1.0,
        show_legend: bool = False,
        show_border: bool = False,
        split_nonnegative_by_corpus: bool = True,
    ) -> alt.LayerChart:
        """
        Basic bar plot
        """
        comp = self._comparator.comparisons[comparison]
        df_plot = self._get_plot_df(
            comparison=comp,
            plot_type="bar",
            max_rank=max_rank,
            split_nonnegative_by_corpus=split_nonnegative_by_corpus,
        )

        return shift(
            df=df_plot,
            label_col=self._comparator._label_col,
            rank_col=RANK_COL,
            score_col=PLOT_CONTRIBUTION_COL,
            score_type_col=PLOT_CONTRIBUTION_TYPE_COL,
            max_rank=max_rank,
            vertical=vertical,
            height=height,
            width=width,
            autoscale_axes=autoscale_axes,
            rank_axis=rank_axis,
            score_axis=InternalScoreAxisConfig.from_config(
                score_axis, all_positive_labels=split_nonnegative_by_corpus and comp.is_nonnegative
            ),
            bar_config=InternalBarConfig.from_config(bar_config, detailed=False),
            label_config=label_config,
            scale_factor=scale_factor,
            show_legend=show_legend,
            show_border=show_border,
        )

    def shift(
        self,
        comparison: str,
        max_rank: int = 50,
        vertical: bool = True,
        height: float | None = None,
        width: float | None = None,
        autoscale_axes: bool = True,
        rank_axis: RankAxisConfig | dict | None = None,
        score_axis: ScoreAxisConfig | dict | None = None,
        bar_config: BarConfig | dict | None = None,
        label_config: LabelConfig | dict | None = None,
        scale_factor: float = 1.0,
        show_legend: bool = False,
        show_border: bool = False,
        split_nonnegative_by_corpus: bool = True,
    ) -> alt.LayerChart:
        """
        Detailed word shift bar plot
        """
        comp = self._comparator.comparisons[comparison]
        if not comp.is_weighted_avg:
            raise ValueError(
                "Cannot make a word shift plot for measures that are not weighted averages"
            )

        df_plot = self._get_plot_df(
            comparison=comp,
            plot_type="shift",
            max_rank=max_rank,
            split_nonnegative_by_corpus=split_nonnegative_by_corpus,
        )

        return shift(
            df=df_plot,
            label_col=self._comparator._label_col,
            rank_col=RANK_COL,
            score_col=PLOT_CONTRIBUTION_COL,
            score_type_col=PLOT_CONTRIBUTION_TYPE_COL,
            max_rank=max_rank,
            vertical=vertical,
            height=height,
            width=width,
            autoscale_axes=autoscale_axes,
            rank_axis=rank_axis,
            score_axis=InternalScoreAxisConfig.from_config(
                score_axis, all_positive_labels=split_nonnegative_by_corpus and comp.is_nonnegative
            ),
            bar_config=InternalBarConfig.from_config(bar_config, detailed=True),
            label_config=label_config,
            scale_factor=scale_factor,
            show_legend=show_legend,
            show_border=show_border,
        )

    def _get_plot_df(
        self,
        comparison: ComparisonResult,
        plot_type: PlotType,
        max_rank: int = 50,
        split_nonnegative_by_corpus: bool = True,
    ) -> pl.DataFrame:
        """ """
        cache_key = PlotCacheKey(
            alias=comparison.alias,
            plot_type=plot_type,
            split_nonnegative_by_corpus=split_nonnegative_by_corpus,
        )

        # Check cache for a previously computed version of the dataframe
        if cache_key in self._plot_dfs:
            if max_rank == self._plot_dfs[cache_key].max_rank:
                return self._plot_dfs[cache_key].df
            elif max_rank < self._plot_dfs[cache_key].max_rank:
                return self._plot_dfs[cache_key].df.filter(col(RANK_COL) <= max_rank)

        # Compute the dataframe if not seen in the cache
        if plot_type == "bar":
            df_plot = bar_plot_df(
                lf=comparison._lf,
                label_col=self._comparator._label_col,
                score_col=SCORE_COL,
                rank_col=RANK_COL,
                attribution_col=ATTRIBUTION_COL,
                contribution_alias=PLOT_CONTRIBUTION_COL,
                contribution_type_alias=PLOT_CONTRIBUTION_TYPE_COL,
                reference_attribution=REFERENCE_ATTRIBUTION,
                comparison_attribution=COMPARISON_ATTRIBUTION,
                max_rank=max_rank,
                is_nonnegative=comparison.is_nonnegative,
                split_nonnegative_by_corpus=split_nonnegative_by_corpus,
            )
        elif plot_type == "shift":
            df_plot = shift_plot_df(
                df=comparison._lf,
                label_col=self._comparator._label_col,
                rank_col=RANK_COL,
                attribution_col=ATTRIBUTION_COL,
                contribution_alias=PLOT_CONTRIBUTION_COL,
                contribution_type_alias=PLOT_CONTRIBUTION_TYPE_COL,
                reference_attribution=REFERENCE_ATTRIBUTION,
                max_rank=max_rank,
                is_nonnegative=comparison.is_nonnegative,
                split_nonnegative_by_corpus=split_nonnegative_by_corpus,
            )
        else:
            raise ValueError(f"The `plot_type` is not a valid value: {plot_type}")

        self._plot_dfs[cache_key] = PlotCachedDataFrame(df=df_plot, max_rank=max_rank)

        return df_plot
