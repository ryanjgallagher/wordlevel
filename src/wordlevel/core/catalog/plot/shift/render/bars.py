from dataclasses import asdict, dataclass, fields

import altair as alt
import polars as pl
from typing_extensions import Self

from wordlevel.configs import (
    DEFAULT_SCORE_AXIS_TITLE,
    PERCENT_SCORE_AXIS_TITLE,
    SYMBOL_SUFFIX,
    BarConfig,
    RankAxisConfig,
    ScoreAxisConfig,
    ScoreSignBarConfig,
    ShiftBarConfig,
    TotalShiftConfig,
)
from wordlevel.core.catalog.plot.shift.render.border import _TRANSLATE_PX, InternalBorderConfig
from wordlevel.core.catalog.plot.utils import to_altair_optional
from wordlevel.core.comparison.score.attribution import ATTRIBUTION_TYPE


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
        config: ScoreAxisConfig | dict | None,
        all_positive_labels: bool,
        percent: bool = False,
    ) -> Self:
        """
        Resolves a user `ScoreAxisConfig` into the internal config. When contributions are shown as
        percentages and the title is left at its default, the title reflects percentages.
        """
        if config is None:
            config = ScoreAxisConfig()
        elif isinstance(config, dict):
            config = ScoreAxisConfig(**config)

        config = asdict(config)
        # Only default to the percent title if the user didn't specify it
        if percent and config["title"] == DEFAULT_SCORE_AXIS_TITLE:
            config["title"] = PERCENT_SCORE_AXIS_TITLE

        return cls(**config, all_positive_labels=all_positive_labels, percent=percent)


@dataclass
class InternalBarConfig:
    """
    Internal configuration of bar colors, stacking orders, and totals symbols.

    Attributes:
        colors:
            Mapping from score type name to bar color
        orders:
            Mapping from score type name to stacking order. `None` when stacking order is not
            applicable (e.g., for a bar plot whose bars are not stacked)
        symbols:
            Mapping from score type name to symbol to use in the "totals" chart
    """

    colors: dict[str, str]
    orders: dict[str, int] | None
    symbols: dict[str, str]

    @classmethod
    def from_bar_config(
        cls,
        config: BarConfig | ScoreSignBarConfig | dict | None,
        attributed_by: ATTRIBUTION_TYPE,
    ) -> Self:
        """
        Creates an InternalBarConfig from a BarConfig or ScoreSignBarConfig.

        The expected config class depends on how the comparison is attributed
        """
        expected_config: type[BarConfig | ScoreSignBarConfig]
        if attributed_by == "score_sign":
            expected_config = ScoreSignBarConfig
        else:
            expected_config = BarConfig

        resolved_config = cls._resolve_config(config, expected_config)
        colors, symbols = cls._get_colors_and_symbols(resolved_config)

        return cls(colors=colors, orders=None, symbols=symbols)

    @classmethod
    def from_shift_bar_config(cls, config: ShiftBarConfig | dict | None) -> Self:
        """
        Creats an InternalBarConfig from a ShiftBarConfig. The key different from `from_bar_config`
        is that it enforces the order of the different bar types
        """
        resolved_config = cls._resolve_config(config, ShiftBarConfig)
        colors, symbols = cls._get_colors_and_symbols(resolved_config)

        return cls(colors=colors, orders=_SHIFT_TYPE_ORDERS, symbols=symbols)

    @classmethod
    def from_score_types(cls, score_types: list[str]) -> Self:
        """
        No config supplied — derive a uniform-color colors dict from whatever score types are
        present in the data. Callers that want plot-type-specific defaults are expected to
        construct an InternalBarConfig via from_bar_config / from_shift_bar_config.
        """
        # TODO: infer if the score types correspond to reference/comparison, reinforce/offset, or
        #       the shift score types, and route to those constructors instead
        return cls(
            colors={st: "steelblue" for st in score_types},
            orders=None,
            symbols={st: st for st in score_types},
        )

    @classmethod
    def from_total_shift_config(cls, config: TotalShiftConfig | dict | None) -> Self:
        """
        Builds the grand-total bar config from a `TotalShiftConfig`, using its `total` color and
        `total_symbol` for the single `"total"` bar.
        """
        resolved_config = cls._resolve_config(config, TotalShiftConfig)
        colors, symbols = cls._get_colors_and_symbols(resolved_config)

        return cls(colors=colors, orders=None, symbols=symbols)

    @staticmethod
    def _resolve_config(
        config: BarConfig | ScoreSignBarConfig | ShiftBarConfig | TotalShiftConfig | dict | None,
        expected_config: (
            type[BarConfig]
            | type[ScoreSignBarConfig]
            | type[ShiftBarConfig]
            | type[TotalShiftConfig]
        ),
    ) -> BarConfig | ScoreSignBarConfig | ShiftBarConfig | TotalShiftConfig:
        """
        Resolves a user-supplied config (which may be None or a dict) and resolves it into the
        corresponding config data class
        """
        if config is None:
            return expected_config()
        if isinstance(config, dict):
            valid_keys = {f.name for f in fields(expected_config)}
            invalid_keys = set(config) - valid_keys
            if invalid_keys:
                raise ValueError(
                    f"Invalid {expected_config.__name__} key(s) {sorted(invalid_keys)}. Valid keys "
                    f"are {sorted(valid_keys)} (or pass a {expected_config.__name__})."
                )
            return expected_config(**config)
        if not isinstance(config, expected_config):
            raise TypeError(
                f"Expected a {expected_config.__name__} or a dict of its fields, but got a "
                f"{type(config).__name__}."
            )
        return config

    @staticmethod
    def _get_colors_and_symbols(
        config: BarConfig | ScoreSignBarConfig | ShiftBarConfig | TotalShiftConfig,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """
        Resolves a config dataclass into its color and symbol mappings (as dicts)
        """
        data = asdict(config)

        colors = {k: v for k, v in data.items() if not k.endswith(SYMBOL_SUFFIX)}
        symbols = {
            k.removesuffix(SYMBOL_SUFFIX): v for k, v in data.items() if k.endswith(SYMBOL_SUFFIX)
        }

        return colors, symbols


def shift_bars(
    df: pl.DataFrame,
    score_col: str,
    rank_col: str,
    score_type_col: str,
    max_rank: int = 50,
    filter_max_rank: bool = True,
    vertical: bool = True,
    height: float | None = None,
    width: float | None = None,
    rank_axis: RankAxisConfig | dict | None = None,
    score_axis: InternalScoreAxisConfig | dict | None = None,
    bars: InternalBarConfig | None = None,
    border: InternalBorderConfig | None = None,
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
        rank_axis:
            Configuration for the rank axis
        score_axis:
            Configuration for the score axis
        bars:
            Configuration for the bars
        border:
            Configuration for the axes and border
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

    # Configure defaults for rank and score axes and the bars
    if isinstance(rank_axis, dict):
        rank_axis = RankAxisConfig(**rank_axis)
    elif rank_axis is None:
        rank_axis = RankAxisConfig()

    if isinstance(score_axis, dict):
        score_axis = InternalScoreAxisConfig(**score_axis)
    elif score_axis is None:
        score_axis = InternalScoreAxisConfig()

    if bars is None:
        score_types = df[score_type_col].unique().to_list()
        bars = InternalBarConfig.from_score_types(score_types)

    if border is None:
        border = InternalBorderConfig()

    # Configure the legend
    if show_legend:
        legend = alt.Undefined
    else:
        # To show the legend, we overwrite that it's "undefined" according to Altair
        legend = None

    # Configure the chart size. We always use `rank_axis.bar_width` to control the size of the bars,
    # implicitly controlling the size of the whole rank axis. If the corresponding dimension param
    # is passed, it is ignored
    # For the configurable dimension, Altair does not allow them to be None or alt.Undefined, so we
    # unpack them from a dict (default is that they're not in the dictionary).
    chart_params = {}
    if vertical:
        chart_params["height"] = alt.Step(rank_axis.bar_width)
        if width is not None:
            chart_params["width"] = width
    else:
        chart_params["width"] = alt.Step(rank_axis.bar_width)
        if height is not None:
            chart_params["height"] = height

    # Suppress the default "border" stroke. We draw borders as axis-domain lines instead
    chart_params["view"] = alt.ViewConfig(strokeWidth=0)

    # Configure the rank axis tick values and labels
    if rank_axis.label_tick_values is not None:
        # Case 1: User passed ticks explicitly
        rank_tick_values = rank_axis.label_tick_values
    elif rank_axis.label_tick_step is not None:
        # Case 2: User followed defaults, i.e. did not pass ticks and did not opt out of tick step
        step = rank_axis.label_tick_step
        ticks = list(range(step, max_rank + 1, step))
        if rank_axis.label_first_tick and (step != 1):
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
    score_type_colors = [bars.colors[score_type] for score_type in score_types]

    # Configure the orders of stacked bars
    if bars.orders is not None:
        df = df.with_columns(pl.col(score_type_col).replace(bars.orders).alias("stacked_bar_order"))
        score_type_orders = alt.Order("stacked_bar_order:Q", sort="ascending")
    else:
        score_type_orders = alt.Undefined

    # Configure the score axis tick labels: optionally force positive values (for non-negative
    # measures shown in both directions) and optionally suffix a percent sign.
    if score_axis.all_positive_labels or score_axis.percent:
        if score_axis.all_positive_labels:
            base = "abs(datum.value)"
        else:
            base = "datum.value"
        suffix = " + '%'" if score_axis.percent else ""
        score_tick_label_expr = base + suffix
    else:
        score_tick_label_expr = alt.Undefined

    # Create aspects of the plot
    score_alt_axis = alt.Axis(
        title=to_altair_optional(score_axis.title),
        titleColor=to_altair_optional(score_axis.title_color),
        titleFontSize=to_altair_optional(score_axis.title_fontsize),
        labelColor=to_altair_optional(score_axis.label_color),
        labelFontSize=to_altair_optional(score_axis.label_fontsize),
        labelExpr=score_tick_label_expr,
        orient="right" if (score_axis_opposite and not vertical) else alt.Undefined,
        grid=score_axis.grid,
        gridColor=to_altair_optional(score_axis.grid_color),
        domain=border.draw_score_axis,
        domainColor=to_altair_optional(
            score_axis.axis_color if score_axis.axis_color is not None else border.color
        ),
        domainWidth=score_axis.width,
        tickColor=to_altair_optional(
            score_axis.tick_color if score_axis.tick_color is not None else border.color
        ),
        translate=_TRANSLATE_PX,
    )
    rank_alt_axis = alt.Axis(
        title=to_altair_optional(rank_axis.title),
        titleColor=to_altair_optional(rank_axis.title_color),
        titleFontSize=to_altair_optional(rank_axis.title_fontsize),
        labelColor=to_altair_optional(rank_axis.label_color),
        labelFontSize=to_altair_optional(rank_axis.label_fontsize),
        values=rank_tick_values,
        labelAngle=rank_label_rotation,
        grid=rank_axis.grid,
        gridColor=to_altair_optional(rank_axis.grid_color),
        domain=border.draw_rank_axis,
        domainColor=to_altair_optional(
            rank_axis.axis_color if rank_axis.axis_color is not None else border.color
        ),
        domainWidth=rank_axis.width,
        tickColor=to_altair_optional(
            rank_axis.tick_color if rank_axis.tick_color is not None else border.color
        ),
        translate=_TRANSLATE_PX,
    )

    # `nice=False` is needed to align ticks correctly with the border
    score_alt_scale = alt.Scale(nice=False)
    rank_alt_scale = alt.Scale(
        paddingOuter=rank_axis.axis_padding, paddingInner=rank_axis.bar_padding
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
        .mark_bar()
        .encode(x=x_axis, y=y_axis, color=color, order=score_type_orders)
        .properties(**chart_params)
    )

    return bars_chart
