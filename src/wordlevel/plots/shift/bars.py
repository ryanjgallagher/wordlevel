import altair as alt
import polars as pl

from typing import Literal
from typing_extensions import Self
from dataclasses import dataclass

from wordlevel.plots.shift.configs import shift_score_type2color, shift_score_type2bar_order
from wordlevel.plots.utils import to_altair_optional


@dataclass
class ScoreAxisConfig:
    """ """

    title: str | None = "Contribution"
    title_fontsize: float | None = None
    label_fontsize: float | None = None
    grid: bool = False


@dataclass
class InternalScoreAxisConfig(ScoreAxisConfig):
    """ """

    all_positive_labels: bool = False

    @classmethod
    def from_config(cls, config: ScoreAxisConfig | dict | None, all_positive_labels: bool) -> Self:
        """ """
        if config is None:
            config = ScoreAxisConfig()
        elif isinstance(config, dict):
            config = ScoreAxisConfig(**config)
        return cls(**config.asdict(), all_positive_labels=all_positive_labels)


@dataclass
class RankAxisConfig:
    """ """

    title: str | None = "Rank"
    title_fontsize: float | None = None
    label_fontsize: float | None = None
    label_tick_values: list[int] | None = None
    label_tick_step: int | None = 5
    label_first_tick: bool = True
    grid: bool = True


@dataclass
class BarConfig:
    """ """

    colors: dict[str, str] | Literal["diverging", "category"] = None
    orders: dict[str, int] | None = None


@dataclass
class InternalBarConfig(BarConfig):
    """ """

    detailed: bool = False

    @classmethod
    def from_config(cls, config: BarConfig | dict | None, detailed: bool) -> Self:
        """ """
        if config is None:
            config = BarConfig()
        elif isinstance(config, dict):
            config = BarConfig(**config)
        return cls(**config.asdict(), detailed=detailed)

    def __post_init__(self):
        """ """
        if self.detailed:
            self.orders = shift_score_type2bar_order

            if self.colors is None:
                self.colors = shift_score_type2color
            else:
                # Override default colors with those that were specified
                self.colors = shift_score_type2color | self.colors
        else:
            if self.colors is None:
                self.colors = "diverging"


def shift_bars(
    df: pl.DataFrame,
    score_col: str,
    rank_col: str,
    score_type_col: str,
    max_rank: int = 50,
    filter_max_rank: bool = True,
    vertical: bool = True,
    height: float = None,
    width: float = None,
    rank_axis: RankAxisConfig | dict | None = None,
    score_axis: InternalScoreAxisConfig | dict | None = None,
    bars: InternalBarConfig | dict | None = None,
    show_legend: bool = False,
    show_border: bool = False,
) -> alt.Chart:
    """
    Assumes a dataframe which has one row per word per type of contribution. There should be columns
    for the contribution, contribution type, and rank
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

    if isinstance(bars, dict):
        bars = InternalBarConfig(**bars)
    elif bars is None:
        bars = InternalBarConfig()

    # Configure whether to show all border axes, i.e. whether to hide the top and right axes
    if show_border:
        axes_view = alt.Undefined
    else:
        axes_view = alt.ViewConfig(strokeWidth=0)

    # Configure the legend
    if show_legend:
        legend = alt.Undefined
    else:
        # To show the legend, we overwrite that it's "undefined" according to Altair
        legend = None

    # Configure the width and height. Altair does not allow these to be None or alt.Undefined.
    # Instead, we have to unpack them from a dict (default is that they're not in the dictionary)
    chart_size = dict()
    if width is not None:
        chart_size["width"] = width
    if height is not None:
        chart_size["height"] = height

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

    # Configure the score types and colors
    if bars.detailed:
        score_types = list(bars.colors.keys())
    else:
        # Get the score types from the data, in case user missed any colors they passed in a dict
        score_types = df[score_type_col].unique().to_list()

    if isinstance(bars.colors, dict):
        score_type_colors = [bars.colors[score_type] for score_type in score_types]
    else:
        score_type_colors = bars.colors

    # Configure the orders of stacked bars
    if bars.orders is not None:
        df = df.with_columns(pl.col(score_type_col).replace(bars.orders).alias("stacked_bar_order"))
        score_type_orders = alt.Order("stacked_bar_order:Q", sort="ascending")
    else:
        score_type_orders = alt.Undefined

    # Configure whether to show all positive labels on the score axis
    if score_axis.all_positive_labels:
        score_label_expr = "abs(datum.value)"
    else:
        score_label_expr = alt.Undefined

    # Create aspects of the plot
    score_alt_axis = alt.Axis(
        title=to_altair_optional(score_axis.title),
        titleFontSize=to_altair_optional(score_axis.title_fontsize),
        labelFontSize=to_altair_optional(score_axis.label_fontsize),
        grid=score_axis.grid,
        labelExpr=score_label_expr,
    )
    rank_alt_axis = alt.Axis(
        title=to_altair_optional(rank_axis.title),
        titleFontSize=to_altair_optional(rank_axis.title_fontsize),
        labelFontSize=to_altair_optional(rank_axis.label_fontsize),
        labelAngle=rank_label_rotation,
        grid=rank_axis.grid,
        values=rank_tick_values,
    )

    color = alt.Color(
        f"{score_type_col}:N",
        scale=alt.Scale(domain=score_types, range=score_type_colors),
        legend=legend,
    )

    # Configure the orientation
    if vertical:
        x_axis = alt.X(score_col, axis=score_alt_axis)
        y_axis = alt.Y(f"{rank_col}:O", axis=rank_alt_axis)
    else:
        x_axis = alt.X(f"{rank_col}:O", axis=rank_alt_axis)
        y_axis = alt.Y(score_col, axis=score_alt_axis)

    # Create chart layer
    bars_chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(x=x_axis, y=y_axis, color=color, order=score_type_orders)
        .properties(**chart_size, view=axes_view)
    )

    return bars_chart
