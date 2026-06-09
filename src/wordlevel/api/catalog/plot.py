from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import altair as alt
import polars as pl
from polars import col

from wordlevel.api.comparison.comparison import ComparisonResult
from wordlevel.constants import (
    ATTRIBUTION_COL,
    PLOT_CONTRIBUTION_COL,
    PLOT_CONTRIBUTION_TYPE_COL,
    RANK_COL,
    SCORE_COL,
    REFERENCE_ATTRIBUTION,
)
from wordlevel.configs import (
    AlignmentBarConfig,
    BarConfig,
    BorderConfig,
    LabelConfig,
    RankAxisConfig,
    ScoreAxisConfig,
    ShiftBarConfig,
)
from wordlevel.core.catalog.plot.shift.transform import bar_plot_df, shift_plot_df
from wordlevel.core.catalog.plot.shift.render.bars import (
    InternalBarConfig,
    InternalScoreAxisConfig,
)
from wordlevel.core.catalog.plot.shift.render.shift import shift

if TYPE_CHECKING:
    from wordlevel.api.catalog.catalog import Catalog


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
    """
    Namespace for plotting comparisons in a `Catalog`.

    Accessed via the `plot` attribute of a `Catalog` (e.g. `catalog.plot.bar(...)`). Each method
    takes the alias of a comparison in the catalog and produces an Altair chart. Underlying plot
    dataframes are cached internally so that repeated calls for the same comparison and plot do
    not do unnecessary recomputation.
    """

    def __init__(self, catalog: Catalog):
        self._catalog = catalog
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
        bar_config: BarConfig | AlignmentBarConfig | dict | None = None,
        label_config: LabelConfig | dict | None = None,
        border_config: BorderConfig | dict | None = None,
        scale_factor: float = 1.0,
        show_legend: bool = False,
        split_nonnegative_by_corpus: bool = True,
    ) -> alt.LayerChart:
        """
        Makes a bar chart of the top contributing tokens distinguishing two compared corpora.

        Bars show the contribution of each token to the overall comparison score, ordered by
        the magnitude. How the contributions are attributed and interpreted depends on the scoring
        measure. For frequency- and rank-based measures—such as
        [proportions][wordlevel.api.comparison.score.Scorer.proportion] and
        [rank-turbulence divergence][wordlevel.api.comparison.score.Scorer.rank_divergence]—each
        token is attributed to the corpus for which it is most characteristic, i.e. which corpus it
        appears more in. For measures where attribution is based on the sign of the
        contribution—such as [lexicon-weighted averages][wordlevel.api.comparison.score.Scorer.lexicon]
        and [Shannon entropy][wordlevel.api.comparison.score.Scorer.shannon_entropy]—a token's
        contribution is interpreted in terms of whether it reinforces or offsets the difference
        between the corpora. See the per-measure scoring documentation for details.

        Example:
            ```python
            --8<-- "snippets/catalog/plot.py:setup"

            --8<-- "snippets/catalog/plot.py:bar"
            ```

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
            | [`Catalog.plot.shift`][wordlevel.api.catalog.plot.Plotter.shift] | Detailed bar plot of comparisons scored with weighted averages |

        Args:
            comparison:
                The alias of a comparison in the catalog. See
                [`Catalog.with_comparisons`][wordlevel.api.catalog.catalog.Catalog.with_comparisons]
                and [`Comparison.alias`][wordlevel.api.comparison.comparison.Comparison.alias]
            max_rank:
                The maximum number of top-contributing tokens to display
            vertical:
                Whether to orient the chart vertically, where tokens are listed top-to-bottom and
                bars extend horizontally. If `False`, tokens are listed left-to-right and bars
                extend vertically
            height:
                The height of the chart, in pixels
            width:
                The width of the chart, in pixels
            autoscale_axes:
                Whether to autoscale the axes to fit the data and the bar labels. If `False`, the
                axes may overlap with the labels
            rank_axis:
                Configuration for the rank axis. Accepts a `RankAxisConfig` or a dict with the same
                fields
            score_axis:
                Configuration for the score axis. Accepts a `ScoreAxisConfig` or a dict with
                the same fields
            bar_config:
                Configuration for the bars. Accepts a `BarConfig`for corpus-attributed measures or
                an `AlignmentBarConfig`for sign-attributed measures. Also accepts a dict keyed by
                the same corresponding attribution labels
            label_config:
                Configuration for the bar labels. Accepts a `LabelConfig` or a dict with the same
                fields
            border_config:
                Configuration for the border of the entire chart. Accepts a `BorderConfig` or a dict
                with the same fields
            scale_factor:
                Multiplier applied to the scale axis when `autoscale_axes` is `True`. Increasing it
                provides more space for the bar labels
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
        comp = self._catalog.comparisons[comparison]
        df_plot = self._get_plot_df(
            comparison=comp,
            plot_type="bar",
            max_rank=max_rank,
            split_nonnegative_by_corpus=split_nonnegative_by_corpus,
        )

        return shift(
            df=df_plot,
            token_col=self._catalog._token_col,
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
                score_axis,
                all_positive_labels=split_nonnegative_by_corpus and comp.is_nonnegative,
                percent=comp.normalized,
            ),
            bar_config=InternalBarConfig.from_bar_config(bar_config, comp.attributed_by),
            label_config=label_config,
            border_config=border_config,
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
        autoscale_axes: bool = True,
        rank_axis: RankAxisConfig | dict | None = None,
        score_axis: ScoreAxisConfig | dict | None = None,
        bar_config: ShiftBarConfig | dict | None = None,
        label_config: LabelConfig | dict | None = None,
        border_config: BorderConfig | dict | None = None,
        scale_factor: float = 1.0,
        show_legend: bool = False,
        split_nonnegative_by_corpus: bool = True,
    ) -> alt.LayerChart:
        r"""
        Makes a word shift plot (a stacked bar chart) of the top contributing tokens distinguishing
        two compared corpora.

        Word shift plots can only be used with scores derived from weighted averages. For comparing
        corpora using measures that are not weighted averages, use
        [`Catalog.plot.bar`][wordlevel.api.catalog.plot.Plotter.bar] instead.

        Example:
            ```python
            --8<-- "snippets/catalog/plot.py:setup"

            --8<-- "snippets/catalog/plot.py:shift"
            ```

        All bars are interpreted in terms of how the comparison corpus differs from the reference
        corpus. For details, see the [scoring
        documentation][wordlevel.api.comparison.score.Scorer.lexicon]. For each token, there are
        two stacked bars. One bar shows whether the token is relatively positive or negative and
        whether it was used more or less in the comparison versus the reference. The other bar shows
        whether the token's score is higher or lower in the comparison versus the reference. Tokens
        are ranked by the magnitude of both stacked bars together.

        Mathematically, the bars are derived from the following formula. Let $\tau$ be a token,
        and let $p_\tau^{(\mathcal{C})}$ and $\phi_\tau^{(\mathcal{C})}$ be the normalized frequency
        and score of $\tau$ in corpus $\mathcal{C}$ respectively. Then the contribution
        $\delta_\tau$ of the token is:

        $$
        \delta_\tau
            =
            \overbrace{
                \biggl(
                    p_\tau^{(C)} - p_\tau^{(R)}
                \biggr)
            }^{
                \uparrow / \downarrow
            }
            \overbrace{
                \biggl[
                    \frac{1}{2}
                    \left(
                        \phi_\tau^{(C)} + \phi_\tau^{(R)} - \Phi^{(\text{ref})}
                    \right)
                \biggr]
            }^{
                + / -
            }
            +
            \underbrace{
                \frac{1}{2}
                \biggl(
                    p_\tau^{(C)} + p_\tau^{(R)}
                \biggr)
                \biggl(
                    \phi_\tau^{(C)} - \phi_\tau^{(R)}
                \biggr)
            }_{
                \bigtriangleup / \bigtriangledown
            }
        $$

        where $R$ and $C$ are the reference and comparison corpora respectively, and
        $\Phi^{(\text{ref})}$ is the user-provided reference score.

        The stacked bars represent the two parts of the summand. The first bar is whether a word is
        relatively positive $(+ / -)$ and appears more or less in the comparison corpus
        $(\uparrow / \downarrow)$. The second bar is whether the score is more or less positive in
        the comparison corpus $(\bigtriangleup / \bigtriangledown)$. Note, the bars may counteract
        one another if the two summand terms are different signs: this is shown visually with
        faded bars in the word shift plot when that is the case.

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
            | [`Catalog.plot.bar`][wordlevel.api.catalog.plot.Plotter.bar] | Concise bar plot of comparisons scored with any measure |

        Args:
            comparison:
                The alias of a comparison in the catalog. See
                [`Catalog.with_comparisons`][wordlevel.api.catalog.catalog.Catalog.with_comparisons]
                and [`Comparison.alias`][wordlevel.api.comparison.comparison.Comparison.alias]
            max_rank:
                The maximum number of top-contributing tokens to display
            vertical:
                Whether to orient the chart vertically, where tokens are listed top-to-bottom and
                bars extend horizontally. If `False`, tokens are listed left-to-right and bars
                extend vertically
            height:
                The height of the chart, in pixels. Ignored if `vertical`, see
                `RankAxisConfig.rank_size` to control the height
            width:
                The width of the chart, in pixels. Ignored if horizontal (`not vertical`), see
                `RankAxisConfig.rank_size` to control the width
            autoscale_axes:
                Whether to autoscale the axes to fit the data and the bar labels. If `False`, the
                axes may overlap with the labels
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
            scale_factor:
                Multiplier applied to the scale axis when `autoscale_axes` is `True`. Increasing it
                provides more space for the bar labels
            show_legend:
                Whether to show the chart legend
            split_nonnegative_by_corpus:
                When the scoring measure is inherently non-negative, whether to direct bar
                contributions in the positive and negative directions based on which corpus they are
                attributed to (the reference or comparison). The score axis is updated to reflect
                that all token-level scores are still positive. If `False`, attributions can still
                be distinguished per corpus by bar color. Ignored for measures that can be negative

        Returns:
            An Altair chart of the word shift plot

        Raises:
            KeyError: If `comparison` is not the alias of a comparison in the catalog
            ValueError: If the comparison's measure is not a weighted average
        """
        comp = self._catalog.comparisons[comparison]
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
            token_col=self._catalog._token_col,
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
                score_axis,
                all_positive_labels=split_nonnegative_by_corpus and comp.is_nonnegative,
                percent=comp.normalized,
            ),
            bar_config=InternalBarConfig.from_shift_bar_config(bar_config),
            label_config=label_config,
            border_config=border_config,
            scale_factor=scale_factor,
            show_legend=show_legend,
        )

    def _get_plot_df(
        self,
        comparison: ComparisonResult,
        plot_type: PlotType,
        max_rank: int = 50,
        split_nonnegative_by_corpus: bool = True,
    ) -> pl.DataFrame:
        """
        For the given `plot_type`, gets the dataframe needed for plotting from `comparison.df`

        Args:
            comparison:
                The comparison to plot
            plot_type:
                The type of plot to make
            max_rank:
                The number of top-contributing tokens to display
            split_nonnegative_by_corpus:
                When the scoring measure is inherently non-negative, whether to direct bar
                contributions in the positive and negative directions based on which corpus they are
                attributed to (the reference or comparison). The score axis is updated to reflect
                that all token-level scores are still positive. If `False`, attributions can still
                be distinguished per corpus by bar color. Ignored for measures that can be negative

        Returns:
            The properly formatted dataframe needed for plotting

        Raises:
            ValueError: If `plot_type` is not recognized
        """
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

        # Normalized contributions are displayed as percentages.
        if comparison.normalized:
            scale = 100.0
        else:
            scale = 1.0

        # Compute the dataframe if not seen in the cache
        if plot_type == "bar":
            df_plot = bar_plot_df(
                lf=comparison._lf,
                token_col=self._catalog._token_col,
                score_col=SCORE_COL,
                rank_col=RANK_COL,
                attribution_col=ATTRIBUTION_COL,
                contribution_alias=PLOT_CONTRIBUTION_COL,
                contribution_type_alias=PLOT_CONTRIBUTION_TYPE_COL,
                attribution_labels=comparison.attribution_labels,
                max_rank=max_rank,
                is_nonnegative=comparison.is_nonnegative,
                split_nonnegative_by_corpus=split_nonnegative_by_corpus,
                scale=scale,
            )
        elif plot_type == "shift":
            df_plot = shift_plot_df(
                df=comparison._lf,
                token_col=self._catalog._token_col,
                rank_col=RANK_COL,
                attribution_col=ATTRIBUTION_COL,
                contribution_alias=PLOT_CONTRIBUTION_COL,
                contribution_type_alias=PLOT_CONTRIBUTION_TYPE_COL,
                reference_attribution=REFERENCE_ATTRIBUTION,
                max_rank=max_rank,
                is_nonnegative=comparison.is_nonnegative,
                split_nonnegative_by_corpus=split_nonnegative_by_corpus,
                scale=scale,
            )
        else:
            raise ValueError(f"The `plot_type` is not a valid value: {plot_type}")

        self._plot_dfs[cache_key] = PlotCachedDataFrame(df=df_plot, max_rank=max_rank)

        return df_plot
