from dataclasses import dataclass, replace
from typing import Literal, TYPE_CHECKING

import altair as alt
import polars as pl
from polars import col

from wordlevel.compare.comparisons.comparison import ComparisonResult
from wordlevel.constants import CONTRIBUTION_COL, CONTRIBUTION_TYPE_COL, RANK_COL, SHIFT_SCORE_TYPES
from wordlevel.plots.shift.bars import RankAxisConfig, ScoreAxisConfig, BarConfig
from wordlevel.plots.shift.labels import LabelConfig
from wordlevel.plots.shift.shift import shift

if TYPE_CHECKING:
    from wordlevel.compare.comparator import CorpusComparator


PlotType = Literal["bar", "shift"]


@dataclass(frozen=True)
class PlotCacheKey:
    alias: str
    plot_type: PlotType


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
    ) -> alt.LayerChart:
        """
        TODO:
        - Pick defaults for height / width, or make autoscale_axes=False
        - Add parameter for whether bars should be mirrored as positive if non-negative metric
        """
        comp = self._comparator.comparisons[comparison]
        df_plot = self._get_plot_df(comparison=comp, max_rank=max_rank, plot_type="bar")

        return shift(
            df=df_plot,
            label_col=self._comparator._label_col,
            rank_col=RANK_COL,
            score_col=CONTRIBUTION_COL,
            score_type_col=CONTRIBUTION_TYPE_COL,
            max_rank=max_rank,
            vertical=vertical,
            height=height,
            width=width,
            autoscale_axes=autoscale_axes,
            rank_axis=rank_axis,
            score_axis=score_axis,
            bar_config=self._process_bar_config(bar_config, detailed=False),
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
    ) -> alt.LayerChart:
        """
        Detailed word shift bar plot

        TODO:
        - Pick defaults for height / width, or make autoscale_axes=False
        - Add parameter for whether bars should be mirrored as positive if non-negative metric
        """
        comp = self._comparator.comparisons[comparison]
        if not comp.is_weighted_avg:
            raise ValueError(
                "Cannot make a word shift plot for measures that are not weighted averages"
            )

        df_plot = self._get_plot_df(comparison=comp, max_rank=max_rank, plot_type="shift")

        return shift(
            df=df_plot,
            label_col=self._comparator._label_col,
            rank_col=RANK_COL,
            score_col=CONTRIBUTION_COL,
            score_type_col=CONTRIBUTION_TYPE_COL,
            max_rank=max_rank,
            vertical=vertical,
            height=height,
            width=width,
            autoscale_axes=autoscale_axes,
            rank_axis=rank_axis,
            score_axis=score_axis,
            bar_config=self._process_bar_config(bar_config, detailed=True),
            label_config=label_config,
            scale_factor=scale_factor,
            show_legend=show_legend,
            show_border=show_border,
        )

    @staticmethod
    def _process_bar_config(config: BarConfig | dict | None, detailed: bool) -> BarConfig:
        """ """
        if config is None:
            config = BarConfig()
        elif isinstance(config, dict):
            config = BarConfig(**config)

        return replace(config, detailed=detailed)

    def _get_plot_df(
        self, comparison: ComparisonResult, plot_type: PlotType, max_rank: int = 50
    ) -> pl.DataFrame:
        """ """
        cache_key = PlotCacheKey(alias=comparison.alias, plot_type=plot_type)

        # Check cache for a previously computed version of the dataframe
        if cache_key in self._plot_dfs:
            if max_rank == self._plot_dfs[cache_key].max_rank:
                return self._plot_dfs[cache_key].df
            elif max_rank < self._plot_dfs[cache_key].max_rank:
                return self._plot_dfs[cache_key].df.filter(col(RANK_COL) <= max_rank)

        # Compute the dataframe if not seen in the cache
        if plot_type == "bar":
            df_plot = self._get_bar_plot_df(df=comparison._lf, max_rank=max_rank)
        elif plot_type == "shift":
            df_plot = self._get_shift_plot_df(df=comparison._lf, max_rank=max_rank)
        else:
            raise ValueError("The `plot_type` is not a valid value")

        self._plot_dfs[cache_key] = PlotCachedDataFrame(df=df_plot, max_rank=max_rank)

        return df_plot

    def _get_bar_plot_df(self, df: pl.LazyFrame, max_rank: int = 50) -> pl.DataFrame:
        """
        TODO:
        - Orient bars correctly based on frequency
        - Unpivot
        """
        raise NotImplementedError("Bar plot df not yet implemented")

    def _get_shift_plot_df(self, df: pl.LazyFrame, max_rank: int = 50) -> pl.DataFrame:
        """ """
        return (
            df.filter(col(RANK_COL) <= max_rank)
            .unpivot(
                on=SHIFT_SCORE_TYPES,
                index=[self._comparator._label_col, RANK_COL],
                value_name=CONTRIBUTION_COL,
                variable_name=CONTRIBUTION_TYPE_COL,
            )
            .collect()
        )
