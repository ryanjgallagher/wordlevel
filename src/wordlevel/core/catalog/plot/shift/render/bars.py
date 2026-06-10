import altair as alt
import polars as pl

from typing_extensions import Self
from dataclasses import asdict, dataclass

from wordlevel.configs import (
    AlignmentBarConfig,
    BarConfig,
    BorderConfig,
    RankAxisConfig,
    ScoreAxisConfig,
    ShiftBarConfig,
    DEFAULT_SCORE_AXIS_TITLE,
    PERCENT_SCORE_AXIS_TITLE,
)
from wordlevel.core.catalog.plot.utils import to_altair_optional
from wordlevel.core.comparison.score.attribution import ATTRIBUTION_LABELS, ATTRIBUTION_TYPE


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
    Internal configuration of bar colors and stacking orders.

    Attributes:
        colors:
            Mapping from score type name to bar color
        orders:
            Mapping from score type name to stacking order. `None` when stacking order is not
            applicable (e.g., for a bar plot whose bars are not stacked)
    """

    colors: dict[str, str]
    orders: dict[str, int] | None

    @classmethod
    def from_bar_config(
        cls,
        config: BarConfig | AlignmentBarConfig | dict | None,
        attributed_by: ATTRIBUTION_TYPE,
    ) -> Self:
        """
        Creates an InternalBarConfig from a BarConfig or AlignmentBarConfig.

        The expected config class depends on how the comparison is attributed: corpus attribution
        (`"frequency"`, `"rank"`) uses a `BarConfig`; score-sign attribution uses an
        `AlignmentBarConfig`. A dict keyed by the attribution labels is also accepted. A config or
        dict that does not match the comparison's attribution raises a `ValueError`.
        """
        expected_cls: type[BarConfig] | type[AlignmentBarConfig]
        if attributed_by == "score_sign":
            expected_cls = AlignmentBarConfig
        else:
            expected_cls = BarConfig

        labels = ATTRIBUTION_LABELS[attributed_by]

        if config is None:
            config = expected_cls()
        elif isinstance(config, dict):
            invalid_keys = set(config) - set(labels)
            if invalid_keys:
                raise ValueError(
                    f"Invalid bar color key(s) {sorted(invalid_keys)} for a comparison attributed "
                    f"by {attributed_by!r}. Valid keys are {list(labels)} (or pass a "
                    f"{expected_cls.__name__})."
                )
            config = expected_cls(**config)
        elif not isinstance(config, expected_cls):
            raise ValueError(
                f"A comparison attributed by {attributed_by!r} expects a {expected_cls.__name__} "
                f"or a dict keyed by {list(labels)}, but got a {type(config).__name__}."
            )

        return cls(colors=asdict(config), orders=None)

    @classmethod
    def from_shift_bar_config(cls, config: ShiftBarConfig | dict | None) -> Self:
        """
        Creats an InternalBarConfig from a ShiftBarConfig. The key different from `from_bar_config`
        is that it enforces the order of the different bar types
        """
        if config is None:
            config = ShiftBarConfig()
        elif isinstance(config, dict):
            config = ShiftBarConfig(**config)
        return cls(colors=asdict(config), orders=_SHIFT_TYPE_ORDERS)


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
    border: BorderConfig | dict | None = None,
    show_legend: bool = False,
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
        # No config supplied — derive a uniform-color colors dict from whatever score types are
        # present in the data. Callers that want plot-type-specific defaults are expected to
        # construct an InternalBarConfig via from_bar_config / from_shift_bar_config.
        bars = InternalBarConfig(
            colors={st: "steelblue" for st in df[score_type_col].unique().to_list()},
            orders=None,
        )

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
    chart_params = dict()
    if vertical:
        chart_params["height"] = alt.Step(rank_axis.bar_width)
        if width is not None:
            chart_params["width"] = width
    else:
        chart_params["width"] = alt.Step(rank_axis.bar_width)
        if height is not None:
            chart_params["height"] = height

    # Configure the border
    if isinstance(border, dict):
        border = BorderConfig(**border)
    elif border is None:
        border = BorderConfig()

    chart_params["view"] = alt.ViewConfig(
        stroke=to_altair_optional(border.color), strokeWidth=border.width
    )

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
    if score_axis.all_positive_labels:
        score_tick_label_expr = "abs(datum.value)"
    else:
        score_tick_label_expr = "datum.value"

    if score_axis.percent:
        score_tick_label_expr = score_tick_label_expr + " + '%'"
    elif score_axis.all_positive_labels:
        score_tick_label_expr = score_tick_label_expr
    else:
        score_tick_label_expr = alt.Undefined

    # Create aspects of the plot
    # translate=0 removes Altair's default half-pixel axis nudge so the axis ticks and gridlines
    # align exactly with the bars and the zero line
    score_alt_axis = alt.Axis(
        title=to_altair_optional(score_axis.title),
        titleColor=to_altair_optional(score_axis.title_color),
        titleFontSize=to_altair_optional(score_axis.title_fontsize),
        labelColor=to_altair_optional(score_axis.label_color),
        labelFontSize=to_altair_optional(score_axis.label_fontsize),
        labelExpr=score_tick_label_expr,
        grid=score_axis.grid,
        gridColor=to_altair_optional(score_axis.grid_color),
        domainColor=to_altair_optional(
            score_axis.axis_color if score_axis.axis_color is not None else border.color
        ),
        tickColor=to_altair_optional(
            score_axis.tick_color if score_axis.tick_color is not None else border.color
        ),
        translate=0,
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
        domainColor=to_altair_optional(
            rank_axis.axis_color if rank_axis.axis_color is not None else border.color
        ),
        tickColor=to_altair_optional(
            rank_axis.tick_color if rank_axis.tick_color is not None else border.color
        ),
        translate=0,
    )
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
        x_axis = alt.X(score_col, axis=score_alt_axis)
        y_axis = alt.Y(f"{rank_col}:O", axis=rank_alt_axis, scale=rank_alt_scale)
    else:
        x_axis = alt.X(f"{rank_col}:O", axis=rank_alt_axis, scale=rank_alt_scale)
        y_axis = alt.Y(score_col, axis=score_alt_axis)

    # Create chart layer
    bars_chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(x=x_axis, y=y_axis, color=color, order=score_type_orders)
        .properties(**chart_params)
    )

    return bars_chart
