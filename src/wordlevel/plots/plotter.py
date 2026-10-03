from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import altair as alt
import polars as pl
from polars import col

from wordlevel.comparison.comparison import ComparisonResult
from wordlevel.configs import (
    BarConfig,
    BorderConfig,
    LabelConfig,
    RankAxisConfig,
    ScoreAxisConfig,
    ScoreSignBarConfig,
    ShiftBarConfig,
    TotalShiftConfig,
    ZeroLineConfig,
)
from wordlevel.constants import (
    ATTRIBUTION_COL,
    IS_BORROWED_COL,
    PLOT_CONTRIBUTION_COL,
    PLOT_CONTRIBUTION_TYPE_COL,
    RANK_COL,
    SCORE_COL,
)
from wordlevel.plots.core.shift.bars import (
    InternalBarConfig,
    InternalScoreAxisConfig,
)
from wordlevel.plots.core.shift.shift import shift
from wordlevel.plots.core.shift.transform import (
    append_borrow_symbol_to_tokens,
    bar_plot_df,
    shift_plot_df,
)


PlotType = Literal["bar", "shift"]


@dataclass(frozen=True)
class PlotCacheKey:
    alias: str
    plot_type: PlotType
    split_nonnegative_by_corpus: bool
    show_totals: bool = False


@dataclass
class PlotCachedDataFrame:
    df: pl.DataFrame
    max_rank: int | None


class Plotter:
    """
    Namespace for plotting comparisons in a `Catalog`.

    Accessed via the `plot` attribute of a `Catalog` (e.g. `catalog.plot.bar(...)`). Each method
    takes the alias of a comparison in the catalog and produces an Altair chart. Underlying plot
    dataframes are cached internally so that repeated calls for the same comparison and plot do
    not do unnecessary recomputation.
    """

    def __init__(self, comparisons: Mapping[str, ComparisonResult], token_col: str):
        self._comparisons = comparisons
        self._token_col = token_col
        self._plot_dfs: dict[PlotCacheKey, PlotCachedDataFrame] = {}

    def bar(
        self,
        comparison: str,
        max_rank: int = 50,
        vertical: bool = True,
        height: float | None = None,
        width: float | None = None,
        show_totals: bool = False,
        rank_axis: RankAxisConfig | dict | None = None,
        score_axis: ScoreAxisConfig | dict | None = None,
        bar_config: BarConfig | ScoreSignBarConfig | dict | None = None,
        label_config: LabelConfig | dict | None = None,
        border_config: BorderConfig | dict | None = None,
        zero_line_config: ZeroLineConfig | dict | None = None,
        total_shift_config: TotalShiftConfig | dict | None = None,
        scale_factor: float = 1.0,
        show_legend: bool = False,
        split_nonnegative_by_corpus: bool = True,
    ) -> alt.LayerChart | alt.VConcatChart | alt.HConcatChart:
        """
        Makes a bar chart of the top contributing tokens distinguishing two compared corpora.

        Example:
            ```python
            --8<-- "snippets/catalog/plot.py:setup"

            --8<-- "snippets/catalog/plot.py:bar"
            ```

        Bars show the contribution of each token to the overall comparison score, ordered by
        magnitude. Which two groups the tokens are split into, and how each side reads, depends on
        the scoring measure.

        Reference:
            For more details, see the following paper:
            > Gallagher, R. J., Frank, M. R., Mitchell, L., Schwartz, A. J., Reagan, A. J.,
            Danforth, C. M., & Dodds, P. S. (2021). [Generalized word shift graphs: a method for
            visualizing and explaining pairwise comparisons
            between texts](https://link.springer.com/article/10.1140/epjds/s13688-021-00260-3).
            EPJ Data Science, 10(1), 4.

        See Also:
            | Plot | Description |
            |------|-------------|
            | [`Catalog.plot.shift`][wordlevel.plots.plotter.Plotter.shift] | Detailed bar plot of comparisons scored with weighted averages |

        Args:
            comparison:
                The alias of a comparison in the catalog. See
                [`Catalog.with_comparisons`][wordlevel.catalog.catalog.Catalog.with_comparisons]
                and [`Comparison.alias`][wordlevel.comparison.comparison.Comparison.alias]
            max_rank:
                The maximum number of top-contributing tokens to display
            vertical:
                Whether to orient the chart vertically, where tokens are listed top-to-bottom and
                bars extend horizontally. If `False`, tokens are listed left-to-right and bars
                extend vertically
            height:
                The height of the chart, in pixels. Ignored if `vertical`, see
                `RankAxisConfig.bar_width` to control the height
            width:
                The width of the chart, in pixels. Ignored if horizontal (`vertical=False`), see
                `RankAxisConfig.bar_width` to control the width
            show_totals:
                Whether to show the overall total contributions of each score type and overall as an
                additional chart appended to the main chart
            rank_axis:
                Configuration for the rank axis. Accepts a `RankAxisConfig` or a dict with the same
                fields
            score_axis:
                Configuration for the score axis. Accepts a `ScoreAxisConfig` or a dict with
                the same fields
            bar_config:
                Configuration for the bars. Accepts a `BarConfig` for corpus-attributed measures or
                a `ScoreSignBarConfig` for sign-attributed measures. Also accepts a dict keyed by
                the same corresponding attribution labels
            label_config:
                Configuration for the bar labels. Accepts a `LabelConfig` or a dict with the same
                fields
            border_config:
                Configuration for the border of the entire chart. Accepts a `BorderConfig` or a dict
                with the same fields
            zero_line_config:
                Configuration for the zero line at the score-axis origin. Accepts a `ZeroLineConfig`
                or a dict with the same fields
            total_shift_config:
                Configuration for the grand total bar in the totals panels. Accepts a
                `TotalShiftConfig` or a dict with the same fields. Ignored if `show_totals` is
                `False`
            scale_factor:
                Multiplier applied to the score axis when fitting it to the bar labels. Increasing
                it provides more space for the bar labels
            show_legend:
                Whether to show the chart legend
            split_nonnegative_by_corpus:
                When the scoring measure is inherently non-negative, whether to direct bar
                contributions in the positive and negative directions based on which corpus they are
                attributed to (the reference or comparison). The score axis is updated to reflect
                that all token-level scores are still positive. If `False`, attributions can still
                be distinguished per corpus by bar color. Ignored for measures that can be negative

        Returns:
            An Altair chart of the bar plot

        Raises:
            KeyError: If `comparison` is not the alias of a comparison in the catalog
        """
        comp = self._comparisons[comparison]

        if label_config is None:
            label_config = LabelConfig()
        elif isinstance(label_config, dict):
            label_config = LabelConfig(**label_config)

        df_plot = self._get_plot_df(
            comparison=comp,
            plot_type="bar",
            max_rank=None if show_totals else max_rank,
            split_nonnegative_by_corpus=split_nonnegative_by_corpus,
            show_totals=show_totals,
            borrow_symbol=label_config.borrow_symbol,
        )

        return shift(
            df=df_plot,
            token_col=self._token_col,
            rank_col=RANK_COL,
            score_col=PLOT_CONTRIBUTION_COL,
            score_type_col=PLOT_CONTRIBUTION_TYPE_COL,
            max_rank=max_rank,
            vertical=vertical,
            height=height,
            width=width,
            show_totals=show_totals,
            rank_axis=rank_axis,
            score_axis=InternalScoreAxisConfig.from_config(
                score_axis,
                all_positive_labels=split_nonnegative_by_corpus and comp.is_nonnegative,
                percent=comp.normalized,
            ),
            bar_config=InternalBarConfig.from_bar_config(bar_config, comp.attributed_by),
            label_config=label_config,
            border_config=border_config,
            zero_line_config=zero_line_config,
            total_shift_config=total_shift_config,
            scale_factor=scale_factor,
            show_legend=show_legend,
        )

    def shift(
        self,
        comparison: str,
        max_rank: int = 50,
        vertical: bool = True,
        height: float | None = None,
        width: float | None = None,
        show_totals: bool = False,
        rank_axis: RankAxisConfig | dict | None = None,
        score_axis: ScoreAxisConfig | dict | None = None,
        bar_config: ShiftBarConfig | dict | None = None,
        label_config: LabelConfig | dict | None = None,
        border_config: BorderConfig | dict | None = None,
        zero_line_config: ZeroLineConfig | dict | None = None,
        total_shift_config: TotalShiftConfig | dict | None = None,
        scale_factor: float = 1.0,
        show_legend: bool = False,
    ) -> alt.LayerChart | alt.VConcatChart | alt.HConcatChart:
        """
        Makes a word shift plot (a stacked bar chart) of the top contributing tokens distinguishing
        two compared corpora.

        Example:
            ```python
            --8<-- "snippets/catalog/plot.py:setup"

            --8<-- "snippets/catalog/plot.py:shift"
            ```

        There are two stacked bars per token: one indicating the difference in relative frequency,
        one indicating the difference in score. Tokens are ranked by the magnitude of both bars
        stacked together. For what each bar means and how to read the plot, see more on
        [word shift graphs](https://wordlevel.dev/documentation/concepts/word_shift_graphs/).

        Word shift graphs can only be used with scores derived from weighted averages. For comparing
        corpora using measures that are not weighted averages, use
        [`Catalog.plot.bar`][wordlevel.plots.plotter.Plotter.bar] instead.

        Reference:
            For more details, see the following paper:
            > Gallagher, R. J., Frank, M. R., Mitchell, L., Schwartz, A. J., Reagan, A. J.,
            Danforth, C. M., & Dodds, P. S. (2021). [Generalized word shift graphs: a method for
            visualizing and explaining pairwise comparisons
            between texts](https://link.springer.com/article/10.1140/epjds/s13688-021-00260-3).
            EPJ Data Science, 10(1), 4.

        See Also:
            | Plot | Description |
            |------|-------------|
            | [`Catalog.plot.bar`][wordlevel.plots.plotter.Plotter.bar] | Concise bar plot of comparisons scored with any measure |

        Args:
            comparison:
                The alias of a comparison in the catalog. See
                [`Catalog.with_comparisons`][wordlevel.catalog.catalog.Catalog.with_comparisons]
                and [`Comparison.alias`][wordlevel.comparison.comparison.Comparison.alias]
            max_rank:
                The maximum number of top-contributing tokens to display
            vertical:
                Whether to orient the chart vertically, where tokens are listed top-to-bottom and
                bars extend horizontally. If `False`, tokens are listed left-to-right and bars
                extend vertically
            height:
                The height of the chart, in pixels. Ignored if `vertical`, see
                `RankAxisConfig.bar_width` to control the height
            width:
                The width of the chart, in pixels. Ignored if horizontal (`not vertical`), see
                `RankAxisConfig.bar_width` to control the width
            show_totals:
                Whether to show the overall total contributions of each score type and overall as an
                additional chart appended to the main chart
            rank_axis:
                Configuration for the rank axis. Accepts a `RankAxisConfig` or a dict with the same
                fields
            score_axis:
                Configuration for the score axis. Accepts a `ScoreAxisConfig` or a dict with
                the same fields
            bar_config:
                Configuration for the bars. Accepts a `ShiftBarConfig` or a dict with the same
                fields
            label_config:
                Configuration for the bar labels. Accepts a `LabelConfig` or a dict with the same
                fields
            border_config:
                Configuration for the border of the entire chart. Accepts a `BorderConfig` or a dict
                with the same fields
            zero_line_config:
                Configuration for the zero line at the score-axis origin. Accepts a `ZeroLineConfig`
                or a dict with the same fields
            total_shift_config:
                Configuration for the grand total bar in the totals panels. Accepts a
                `TotalShiftConfig` or a dict with the same fields. Ignored if `show_totals` is
                `False`
            scale_factor:
                Multiplier applied to the score axis when fitting it to the bar labels. Increasing
                it provides more space for the bar labels
            show_legend:
                Whether to show the chart legend

        Returns:
            An Altair chart of the word shift plot

        Raises:
            KeyError: If `comparison` is not the alias of a comparison in the catalog
            ValueError: If the comparison's measure is not a weighted average
        """
        comp = self._comparisons[comparison]
        if not comp.is_weighted_avg:
            raise ValueError(
                "Cannot make a word shift plot for measures that are not weighted averages"
            )

        if label_config is None:
            label_config = LabelConfig()
        elif isinstance(label_config, dict):
            label_config = LabelConfig(**label_config)

        df_plot = self._get_plot_df(
            comparison=comp,
            plot_type="shift",
            max_rank=None if show_totals else max_rank,
            split_nonnegative_by_corpus=False,
            show_totals=show_totals,
            borrow_symbol=label_config.borrow_symbol,
        )

        return shift(
            df=df_plot,
            token_col=self._token_col,
            rank_col=RANK_COL,
            score_col=PLOT_CONTRIBUTION_COL,
            score_type_col=PLOT_CONTRIBUTION_TYPE_COL,
            max_rank=max_rank,
            vertical=vertical,
            height=height,
            width=width,
            show_totals=show_totals,
            rank_axis=rank_axis,
            score_axis=InternalScoreAxisConfig.from_config(
                score_axis,
                all_positive_labels=False,
                percent=comp.normalized,
            ),
            bar_config=InternalBarConfig.from_shift_bar_config(bar_config),
            label_config=label_config,
            border_config=border_config,
            zero_line_config=zero_line_config,
            total_shift_config=total_shift_config,
            scale_factor=scale_factor,
            show_legend=show_legend,
        )

    def _get_plot_df(
        self,
        comparison: ComparisonResult,
        plot_type: PlotType,
        max_rank: int | None = 50,
        split_nonnegative_by_corpus: bool = True,
        show_totals: bool = False,
        borrow_symbol: str | None = None,
    ) -> pl.DataFrame:
        """
        For the given `plot_type`, gets the dataframe needed for plotting from `comparison.df`

        Args:
            comparison:
                The comparison to plot
            plot_type:
                The type of plot to make
            max_rank:
                The number of top-contributing tokens to display, or `None` to keep the full
                vocabulary (e.g. so totals can be summed over every token)
            split_nonnegative_by_corpus:
                When the scoring measure is inherently non-negative, whether to direct bar
                contributions in the positive and negative directions based on which corpus they are
                attributed to (the reference or comparison). The score axis is updated to reflect
                that all token-level scores are still positive. If `False`, attributions can still
                be distinguished per corpus by bar color. Ignored for measures that can be negative
            show_totals:
                Whether the plot shows total contributions per score type. Used to key the cache, so
                a full-vocabulary df (totals shown) and a top-`max_rank` df (totals hidden) are
                cached separately

        Returns:
            The properly formatted dataframe needed for plotting

        Raises:
            ValueError: If `plot_type` is not recognized
        """
        cache_key = PlotCacheKey(
            alias=comparison.alias,
            plot_type=plot_type,
            split_nonnegative_by_corpus=split_nonnegative_by_corpus,
            show_totals=show_totals,
        )

        df_plot = self._get_cached_plot_df(cache_key=cache_key, max_rank=max_rank)

        # Compute the dataframe if not seen in the cache
        if df_plot is None:
            # If the scores have been normalized, they're plotted as percentages (x 100)
            if comparison.normalized:
                scale = 100.0
            else:
                scale = 1.0

            # If the scores have been borrowed, that indicator needs to be in the plotting dataframe
            if IS_BORROWED_COL in comparison._lf.collect_schema().names():
                unpivot_passthrough_cols = [IS_BORROWED_COL]
            else:
                unpivot_passthrough_cols = None

            if plot_type == "bar":
                df_plot = bar_plot_df(
                    lf=comparison._lf,
                    token_col=self._token_col,
                    score_col=SCORE_COL,
                    rank_col=RANK_COL,
                    attribution_col=ATTRIBUTION_COL,
                    contribution_alias=PLOT_CONTRIBUTION_COL,
                    contribution_type_alias=PLOT_CONTRIBUTION_TYPE_COL,
                    attribution_labels=comparison.attribution_labels,
                    max_rank=max_rank,
                    is_nonnegative=comparison.is_nonnegative,
                    split_nonnegative_by_corpus=split_nonnegative_by_corpus,
                    passthrough_cols=unpivot_passthrough_cols,
                    scale=scale,
                )
            elif plot_type == "shift":
                df_plot = shift_plot_df(
                    df=comparison._lf,
                    token_col=self._token_col,
                    rank_col=RANK_COL,
                    contribution_alias=PLOT_CONTRIBUTION_COL,
                    contribution_type_alias=PLOT_CONTRIBUTION_TYPE_COL,
                    max_rank=max_rank,
                    passthrough_cols=unpivot_passthrough_cols,
                    scale=scale,
                )
            else:
                raise ValueError(f"The `plot_type` is not a valid value: {plot_type}")

            self._plot_dfs[cache_key] = PlotCachedDataFrame(df=df_plot, max_rank=max_rank)

        # Append symbols after caching, so the cache can still be used even if the symbols change
        df_plot = append_borrow_symbol_to_tokens(
            df=df_plot,
            token_col=self._token_col,
            borrow_col=IS_BORROWED_COL,
            borrow_symbol=borrow_symbol,
        )

        return df_plot

    def _get_cached_plot_df(
        self, cache_key: PlotCacheKey, max_rank: int | None
    ) -> pl.DataFrame | None:
        """
        Returns a plot dataframe if it has already been calculated and cached. Returns None if the
        dataframe has not been cached
        """
        if cache_key in self._plot_dfs:
            cached_data = self._plot_dfs[cache_key]

            # The exact same data has already been cached
            if max_rank == cached_data.max_rank:
                return cached_data.df

            # The cached data can be used if the full dataframe has already been realized, or a
            # version of it with a higher max rank has been
            is_full_df = cached_data.max_rank is None
            is_higher_max_rank = (
                max_rank is not None
                and cached_data.max_rank is not None
                and max_rank < cached_data.max_rank
            )
            can_use_cached_data = is_full_df or is_higher_max_rank
            if max_rank is not None and can_use_cached_data:
                return cached_data.df.filter(col(RANK_COL) <= max_rank)

        return None
