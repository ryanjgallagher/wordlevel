from dataclasses import asdict, dataclass, fields
from typing import TypeVar

import altair as alt
import polars as pl
from typing_extensions import Self

from wordlevel.comparison.core.score.attribution import ATTRIBUTION_TYPE
from wordlevel.configs import (
    DEFAULT_BAR_OUTLINE_COLOR,
    DEFAULT_BAR_OUTLINE_WIDTH,
    DEFAULT_SCORE_AXIS_TITLE,
    NON_COLOR_FIELDS,
    PERCENT_SCORE_AXIS_TITLE,
    SYMBOL_SUFFIX,
    BarConfig,
    RankAxisConfig,
    ScoreAxisConfig,
    ScoreSignBarConfig,
    ShiftBarConfig,
    TotalShiftConfig,
)
from wordlevel.constants import COUNTERACTED_SUFFIX
from wordlevel.plots.core.shift.border import _TRANSLATE_PX, InternalBorderConfig
from wordlevel.plots.core.utils import to_altair_optional


_BarConfigT = TypeVar(
    "_BarConfigT", bound=BarConfig | ScoreSignBarConfig | ShiftBarConfig | TotalShiftConfig
)

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
    Internal configuration of bar colors, stacking orders, totals symbols, and outlines.

    Attributes:
        colors:
            Mapping from score type name to bar color
        orders:
            Mapping from score type name to stacking order. `None` when stacking order is not
            applicable (e.g., for a bar plot whose bars are not stacked)
        symbols:
            Mapping from score type name to symbol to use in the "totals" chart
        opacities:
            Mapping from score type name to bar opacity. `None` when every bar is opaque
        outline_color:
            Color of the outline around each bar. If `None`, Altair's default is used
        outline_width:
            Width of the outline around each bar, in pixels. If `None`, Altair's default is used
    """

    colors: dict[str, str]
    orders: dict[str, int] | None
    symbols: dict[str, str]
    opacities: dict[str, float] | None = None
    outline_color: str | None = DEFAULT_BAR_OUTLINE_COLOR
    outline_width: float | None = DEFAULT_BAR_OUTLINE_WIDTH

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

        return cls(
            colors=colors,
            orders=None,
            symbols=symbols,
            outline_color=resolved_config.outline_color,
            outline_width=resolved_config.outline_width,
        )

    @classmethod
    def from_shift_bar_config(cls, config: ShiftBarConfig | dict | None) -> Self:
        """
        Creates an InternalBarConfig from a ShiftBarConfig. The key difference from
        `from_bar_config` is that it enforces the order of the different bar types
        """
        resolved_config = cls._resolve_config(config, ShiftBarConfig)
        base_colors, symbols = cls._get_colors_and_symbols(resolved_config)

        # The config declares only the six base colors, but we need the counteracted ones too
        colors: dict[str, str] = {}
        opacities: dict[str, float] = {}
        for score_type, color in base_colors.items():
            counteracted = f"{score_type}{COUNTERACTED_SUFFIX}"

            colors[score_type] = color
            colors[counteracted] = color
            opacities[score_type] = 1.0
            opacities[counteracted] = resolved_config.counteracted_opacity

        return cls(
            colors=colors,
            orders=_SHIFT_TYPE_ORDERS,
            symbols=symbols,
            opacities=opacities,
            outline_color=resolved_config.outline_color,
            outline_width=resolved_config.outline_width,
        )

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

        return cls(
            colors=colors,
            orders=None,
            symbols=symbols,
            outline_color=resolved_config.outline_color,
            outline_width=resolved_config.outline_width,
        )

    @staticmethod
    def _resolve_config(
        config: BarConfig | ScoreSignBarConfig | ShiftBarConfig | TotalShiftConfig | dict | None,
        expected_config: type[_BarConfigT],
    ) -> _BarConfigT:
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

        colors = {
            k: v
            for k, v in data.items()
            if not k.endswith(SYMBOL_SUFFIX) and k not in NON_COLOR_FIELDS
        }
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
    rank_axis_config: RankAxisConfig | dict | None = None,
    score_axis_config: InternalScoreAxisConfig | dict | None = None,
    bar_config: InternalBarConfig | None = None,
    border_config: InternalBorderConfig | None = None,
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
        rank_axis_config:
            Configuration for the rank axis
        score_axis_config:
            Configuration for the score axis
        bar_config:
            Configuration for the bars
        border_config:
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
    if isinstance(rank_axis_config, dict):
        rank_axis_config = RankAxisConfig(**rank_axis_config)
    elif rank_axis_config is None:
        rank_axis_config = RankAxisConfig()

    if isinstance(score_axis_config, dict):
        score_axis_config = InternalScoreAxisConfig(**score_axis_config)
    elif score_axis_config is None:
        score_axis_config = InternalScoreAxisConfig()

    if bar_config is None:
        score_types = df[score_type_col].unique().to_list()
        bar_config = InternalBarConfig.from_score_types(score_types)

    if border_config is None:
        border_config = InternalBorderConfig()

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
        fill_opacity = alt.FillOpacity(
            f"{score_type_col}:N",
            scale=alt.Scale(domain=score_types, range=score_type_opacities),
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
