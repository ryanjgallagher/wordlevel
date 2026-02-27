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
    SHIFT_SCORE_TYPES,
)
from wordlevel.expressions.attribution import COMPARISON_ATTRIBUTION, REFERENCE_ATTRIBUTION
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
        split_by_corpus: bool = True,
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
            score_col=PLOT_CONTRIBUTION_COL,
            score_type_col=PLOT_CONTRIBUTION_TYPE_COL,
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
        split_by_corpus: bool = True,
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
            score_col=PLOT_CONTRIBUTION_COL,
            score_type_col=PLOT_CONTRIBUTION_TYPE_COL,
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
    def _process_bar_config(config: BarConfig | dict | None, detailed: bool) -> InternalBarConfig:
        """ """
        if config is None:
            config = BarConfig()
        elif isinstance(config, dict):
            config = BarConfig(**config)

        return InternalBarConfig(**config.asdict(), detailed=detailed)

    def _process_score_axis_config(
        config: ScoreAxisConfig | dict | None, all_positive_labels: bool
    ) -> InternalScoreAxisConfig:
        """ """
        if config is None:
            config = ScoreAxisConfig()
        elif isinstance(config, dict):
            config = ScoreAxisConfig(**config)

        return InternalScoreAxisConfig(**config.asdict(), all_positive_labels=all_positive_labels)

    def _get_plot_df(
        self,
        comparison: ComparisonResult,
        plot_type: PlotType,
        max_rank: int = 50,
        split_by_corpus: bool = True,
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
            df_plot = self._get_bar_plot_df(
                comparison=comparison, max_rank=max_rank, split_by_corpus=split_by_corpus
            )
        elif plot_type == "shift":
            df_plot = self._get_shift_plot_df(df=comparison._lf, max_rank=max_rank)
        else:
            raise ValueError("The `plot_type` is not a valid value")

        self._plot_dfs[cache_key] = PlotCachedDataFrame(df=df_plot, max_rank=max_rank)

        return df_plot

    def _get_bar_plot_df(
        self, comparison: ComparisonResult, max_rank: int = 50, split_by_corpus: bool = True
    ) -> pl.DataFrame:
        """
        TODO: no magic strings
        """
        lf_plot = comparison._lf.filter(col(RANK_COL) <= max_rank)

        # Handle non-negative scores and splitting by corpus
        ref_score = SCORE_COL
        cmp_score = SCORE_COL
        if comparison.is_nonnegative:
            if split_by_corpus and comparison.is_attributed:
                ref_score = -1 * SCORE_COL
                cmp_score = SCORE_COL
            elif split_by_corpus:
                raise ValueError(
                    f"Cannot split by corpus without attributions: {comparison.method}"
                )

        # Attribute contributions to each corpus
        if comparison.is_attributed:
            lf_plot = lf_plot.with_columns(
                pl.when(col(ATTRIBUTION_COL) == REFERENCE_ATTRIBUTION)
                .then(ref_score)
                .otherwise(0)
                .alias("reference"),
                pl.when(col(ATTRIBUTION_COL) == COMPARISON_ATTRIBUTION)
                .then(cmp_score)
                .otherwise(0)
                .alias("comparison"),
            )
        else:
            lf_plot = lf_plot.with_columns(
                pl.when(col(SCORE_COL) < 0).then(ref_score).otherwise(0).alias("reference"),
                pl.when(col(SCORE_COL) >= 0).then(cmp_score).otherwise(0).alias("comparison"),
            )

        # Unpivot
        lf_plot = lf_plot.unpivot(
            on=["reference", "comparison"],
            index=[self._comparator._label_col, RANK_COL],
            value_name=PLOT_CONTRIBUTION_COL,
            variable_name=PLOT_CONTRIBUTION_TYPE_COL,
        )

        return lf_plot.collect()

    def _get_shift_plot_df(self, comparison: ComparisonResult, max_rank: int = 50) -> pl.DataFrame:
        """ """
        return (
            comparison._lf.filter(col(RANK_COL) <= max_rank)
            .unpivot(
                on=SHIFT_SCORE_TYPES,
                index=[self._comparator._label_col, RANK_COL],
                value_name=PLOT_CONTRIBUTION_COL,
                variable_name=PLOT_CONTRIBUTION_TYPE_COL,
            )
            .collect()
        )
