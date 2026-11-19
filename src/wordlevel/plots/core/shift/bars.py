from dataclasses import asdict, dataclass
from typing import Self

import altair as alt
import polars as pl

from wordlevel.configs import (
    DEFAULT_ATTRIBUTION_COLORS,
    DEFAULT_SCORE_AXIS_TITLE,
    NON_SCORE_TYPE_FIELDS,
    PERCENT_SCORE_AXIS_TITLE,
    BarColorConfig,
    BarConfig,
    RankAxisConfig,
    ScoreAxisConfig,
    ShiftColorConfig,
)
from wordlevel.constants import COUNTERACTED_SUFFIX, GRAND_TOTAL_SCORE_TYPE
from wordlevel.plots.core.shift.border import _TRANSLATE_PX, InternalBorderConfig
from wordlevel.plots.core.utils import to_altair_optional


_SHIFT_TYPE_ORDERS: dict[str, int] = {
    "more_of_negative": 1,
    "more_of_positive": 1,
    "less_of_positive": 1,
    "less_of_negative": 1,
    "lower_score": 2,
    "higher_score": 2,
    "lower_score_counteracted": 3,
    "higher_score_counteracted": 3,
    "more_of_negative_counteracted": 3,
    "more_of_positive_counteracted": 3,
    "less_of_positive_counteracted": 3,
    "less_of_negative_counteracted": 3,
}


@dataclass
class InternalScoreAxisConfig(ScoreAxisConfig):
    """
    Internal configuration of the ScoreAxisConfig. Inherits from ScoreAxisConfig

    Attributes:
        all_positive_labels:
            Whether the axis tick labels should be all positive values. Used for plotting inherently
            non-negative measures in the positive and negative direction while indicating their
            positive values
        percent:
            Whether the contributions are shown as percentages. When `True`, tick labels are
            suffixed with `%` and the default axis title reflects percentages
    """

    all_positive_labels: bool = False
    percent: bool = False

    @classmethod
    def from_config(
        cls,
        config: ScoreAxisConfig,
        all_positive_labels: bool,
        percent: bool = False,
    ) -> Self:
        """
        Derives the internal config from a user `ScoreAxisConfig`. When contributions are shown as
        percentages and the title is left at its default, the title reflects percentages.
        """
        config_fields = asdict(config)
        # Only default to the percent title if the user didn't specify it
        if percent and config_fields["title"] == DEFAULT_SCORE_AXIS_TITLE:
            config_fields["title"] = PERCENT_SCORE_AXIS_TITLE

        return cls(**config_fields, all_positive_labels=all_positive_labels, percent=percent)


@dataclass
class InternalBarConfig(BarConfig):
    """
    Internal configuration of bar colors, stacking orders, and opacities. Inherits every bar's
    outline from `BarConfig`.

    Each `colors` mapping also carries the grand total bar, so a totals panel can be drawn from
    the same config as the bars it totals.

    Attributes:
        colors:
            Mapping from score type name to bar color
        orders:
            Mapping from score type name to stacking order. `None` when stacking order is not
            applicable (e.g., for a bar plot whose bars are not stacked)
        opacities:
            Mapping from score type name to bar opacity. `None` when every bar is opaque
    """

    colors: dict[str, str]
    orders: dict[str, int] | None
    opacities: dict[str, float] | None = None

    @classmethod
    def from_bar_colors(
        cls,
        config: BarColorConfig,
        bar_config: BarConfig,
        attribution_labels: tuple[str, str],
    ) -> Self:
        """
        Builds the config for a bar plot, coloring the pair of bars that the comparison's
        attribution produces. A color left unset takes its default
        """
        colors = {GRAND_TOTAL_SCORE_TYPE: config.grand_total}
        for label in attribution_labels:
            color = getattr(config, label)
            colors[label] = DEFAULT_ATTRIBUTION_COLORS[label] if color is None else color

        return cls(
            colors=colors,
            orders=None,
            outline_color=bar_config.outline_color,
            outline_width=bar_config.outline_width,
        )

    @classmethod
    def from_shift_colors(
        cls,
        config: ShiftColorConfig,
        bar_config: BarConfig,
    ) -> Self:
        """
        Builds the config for a word shift plot. Unlike `from_bar_colors` it enforces the order of
        the different bar types, and fades each `*_counteracted` bar behind the bar it counteracts
        """
        colors = {GRAND_TOTAL_SCORE_TYPE: config.grand_total}
        opacities: dict[str, float] = {}

        # The config declares only the six base colors, but we need the counteracted ones too
        for score_type, color in cls._get_colors(config).items():
            counteracted = f"{score_type}{COUNTERACTED_SUFFIX}"

            colors[score_type] = color
            colors[counteracted] = color
            opacities[score_type] = 1.0
            opacities[counteracted] = config.counteracted_opacity

        return cls(
            colors=colors,
            orders=_SHIFT_TYPE_ORDERS,
            opacities=opacities,
            outline_color=bar_config.outline_color,
            outline_width=bar_config.outline_width,
        )

    @staticmethod
    def _get_colors(config: ShiftColorConfig) -> dict[str, str]:
        """
        Resolves a color config dataclass into its score type to color mapping
        """
        return {k: v for k, v in asdict(config).items() if k not in NON_SCORE_TYPE_FIELDS}


def shift_bars(
    df: pl.DataFrame,
    score_col: str,
    rank_col: str,
    score_type_col: str,
    rank_axis_config: RankAxisConfig,
    score_axis_config: InternalScoreAxisConfig,
    bar_config: InternalBarConfig,
    border_config: InternalBorderConfig,
    max_rank: int = 50,
    filter_max_rank: bool = True,
    vertical: bool = True,
    height: float | None = None,
    width: float | None = None,
    show_legend: bool = False,
    score_axis_opposite: bool = False,
) -> alt.Chart:
    """
    Makes the bar chart layer of a word shift plot

    Args:
        df:
            A long dataframe with one row per token per type of score to plot. For example, if a
            token has two contributing components `more_of_positive` and `higher_score`, then it
            should have two rows, where the `score_col` indicates the length of each
        score_col:
            Column in the dataframe indicating the "score" associated with each bar, i.e. its length
        rank_col:
            Column in the dataframe indicating the rank of each token, taking into account all of
            its contributing scores
        score_type_col:
            Column in the dataframe indicating the type of "score" associated with each bar
        rank_axis_config:
            Configuration for the rank axis
        score_axis_config:
            Configuration for the score axis
        bar_config:
            Configuration for the bars
        border_config:
            Configuration for the axes and border
        max_rank:
            The maximum number of top-contributing tokens to display
        filter_max_rank:
            Whether to filter by the `max_rank` or ignore the parameter
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
        show_legend:
            Whether to show the chart legend
        score_axis_opposite:
            Whether to place the score axis (labels, ticks, title) on the side opposite its default.
            Ignored if `vertical` is True

    Returns:
        An Altair chart of the bar graph layer
    """
    # Filter the dataframe to the top N rows
    if filter_max_rank:
        df = df.filter(pl.col(rank_col) <= max_rank)

    # Configure the legend
    if show_legend:
        legend = alt.Undefined
    else:
        # To show the legend, we overwrite that it's "undefined" according to Altair
        legend = None

    # Configure the chart size. We always use `rank_axis_config.bar_width` to control the size of the bars,
    # implicitly controlling the size of the whole rank axis. If the corresponding dimension param
    # is passed, it is ignored
    # For the configurable dimension, Altair does not allow them to be None or alt.Undefined, so we
    # unpack them from a dict (default is that they're not in the dictionary).
    chart_params = {}
    if vertical:
        chart_params["height"] = alt.Step(rank_axis_config.bar_width)
        if width is not None:
            chart_params["width"] = width
    else:
        chart_params["width"] = alt.Step(rank_axis_config.bar_width)
        if height is not None:
            chart_params["height"] = height

    # Suppress the default "border" stroke. We draw borders as axis-domain lines instead
    chart_params["view"] = alt.ViewConfig(strokeWidth=0)

    # Configure the rank axis tick values and labels
    if rank_axis_config.label_tick_values is not None:
        # Case 1: User passed ticks explicitly
        rank_tick_values = rank_axis_config.label_tick_values
    elif rank_axis_config.label_tick_step is not None:
        # Case 2: User followed defaults, i.e. did not pass ticks and did not opt out of tick step
        step = rank_axis_config.label_tick_step
        ticks = list(range(step, max_rank + 1, step))
        if rank_axis_config.label_first_tick and (step != 1):
            ticks = [1] + ticks
        rank_tick_values = ticks
    else:
        # Case 3: User wants to see default Altair labels, i.e. opted out of the tick step
        rank_tick_values = alt.Undefined

    if vertical:
        rank_label_rotation = alt.Undefined
    else:
        rank_label_rotation = 0

    # Discover score types from the data and look up colors by name. Sorted for a deterministic
    # alphabetical legend order across runs.
    score_types = sorted(df[score_type_col].unique().to_list())
    score_type_colors = [bar_config.colors[score_type] for score_type in score_types]

    if bar_config.opacities is not None:
        score_type_opacities = [bar_config.opacities[score_type] for score_type in score_types]
        # `type` must be set: opacity is a continuous channel, so Vega-Lite otherwise builds a
        # `point` scale, which spreads the score types evenly between the first and last opacity
        # instead of giving each the one it is paired with
        fill_opacity = alt.FillOpacity(
            f"{score_type_col}:N",
            scale=alt.Scale(type="ordinal", domain=score_types, range=score_type_opacities),
            legend=None,
        )
    else:
        fill_opacity = alt.Undefined

    # Configure the orders of stacked bars
    if bar_config.orders is not None:
        df = df.with_columns(
            pl.col(score_type_col).replace(bar_config.orders).alias("stacked_bar_order")
        )
        score_type_orders = alt.Order("stacked_bar_order:Q", sort="ascending")
    else:
        score_type_orders = alt.Undefined

    # Configure the score axis tick labels: optionally force positive values (for non-negative
    # measures shown in both directions) and optionally suffix a percent sign.
    if score_axis_config.all_positive_labels or score_axis_config.percent:
        if score_axis_config.all_positive_labels:
            base = "abs(datum.value)"
        else:
            base = "datum.value"
        suffix = " + '%'" if score_axis_config.percent else ""
        score_tick_label_expr = base + suffix
    else:
        score_tick_label_expr = alt.Undefined

    # Create aspects of the plot
    score_alt_axis = alt.Axis(
        title=score_axis_config.title,
        titleColor=to_altair_optional(score_axis_config.title_color),
        titleFontSize=to_altair_optional(score_axis_config.title_fontsize),
        labelColor=to_altair_optional(score_axis_config.label_color),
        labelFontSize=to_altair_optional(score_axis_config.label_fontsize),
        labelExpr=score_tick_label_expr,
        orient="right" if (score_axis_opposite and not vertical) else alt.Undefined,
        grid=score_axis_config.grid,
        gridColor=to_altair_optional(score_axis_config.grid_color),
        domain=border_config.draw_score_axis,
        domainColor=to_altair_optional(
            score_axis_config.axis_color
            if score_axis_config.axis_color is not None
            else border_config.color
        ),
        domainWidth=score_axis_config.width,
        tickColor=to_altair_optional(
            score_axis_config.tick_color
            if score_axis_config.tick_color is not None
            else border_config.color
        ),
        translate=_TRANSLATE_PX,
    )
    rank_alt_axis = alt.Axis(
        title=rank_axis_config.title,
        titleColor=to_altair_optional(rank_axis_config.title_color),
        titleFontSize=to_altair_optional(rank_axis_config.title_fontsize),
        labelColor=to_altair_optional(rank_axis_config.label_color),
        labelFontSize=to_altair_optional(rank_axis_config.label_fontsize),
        values=rank_tick_values,
        labelAngle=rank_label_rotation,
        grid=rank_axis_config.grid,
        gridColor=to_altair_optional(rank_axis_config.grid_color),
        domain=border_config.draw_rank_axis,
        domainColor=to_altair_optional(
            rank_axis_config.axis_color
            if rank_axis_config.axis_color is not None
            else border_config.color
        ),
        domainWidth=rank_axis_config.width,
        tickColor=to_altair_optional(
            rank_axis_config.tick_color
            if rank_axis_config.tick_color is not None
            else border_config.color
        ),
        translate=_TRANSLATE_PX,
    )

    # `nice=False` is needed to align ticks correctly with the border
    score_alt_scale = alt.Scale(nice=False)
    rank_alt_scale = alt.Scale(
        paddingOuter=rank_axis_config.axis_padding, paddingInner=rank_axis_config.bar_padding
    )

    color = alt.Color(
        f"{score_type_col}:N",
        scale=alt.Scale(domain=score_types, range=to_altair_optional(score_type_colors)),
        legend=legend,
    )

    # Configure the orientation
    if vertical:
        x_axis = alt.X(f"{score_col}:Q", axis=score_alt_axis, scale=score_alt_scale)
        y_axis = alt.Y(f"{rank_col}:O", axis=rank_alt_axis, scale=rank_alt_scale)
    else:
        x_axis = alt.X(f"{rank_col}:O", axis=rank_alt_axis, scale=rank_alt_scale)
        y_axis = alt.Y(f"{score_col}:Q", axis=score_alt_axis, scale=score_alt_scale)

    # Create chart layer
    bars_chart = (
        alt.Chart(df)
        .mark_bar(
            stroke=to_altair_optional(bar_config.outline_color),
            strokeWidth=to_altair_optional(bar_config.outline_width),
        )
        .encode(
            x=x_axis,
            y=y_axis,
            color=color,
            fillOpacity=fill_opacity,
            order=score_type_orders,
        )
        .properties(**chart_params)
    )

    return bars_chart
