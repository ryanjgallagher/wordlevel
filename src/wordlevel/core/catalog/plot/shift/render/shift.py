from typing import cast

import altair as alt
import polars as pl

from wordlevel.configs import BorderConfig, LabelConfig, RankAxisConfig, ZeroLineConfig
from wordlevel.core.catalog.plot.shift.render.bars import (
    shift_bars,
    InternalScoreAxisConfig,
    InternalBarConfig,
)
from wordlevel.core.catalog.plot.shift.render.labels import shift_labels
from wordlevel.core.catalog.plot.shift.render.scales import shift_scales
from wordlevel.core.catalog.plot.utils import to_altair_optional


DEFAULT_SCORE_AXIS_PIXEL_SIZE = 400


def zero_line(
    config: ZeroLineConfig | dict | None = None,
    border: BorderConfig | dict | None = None,
    vertical: bool = True,
) -> alt.Chart:
    """
    Plots a guide line at `0` on the score axis.

    The line's color comes from `config`, falling back to `border` so it matches the chart frame
    when not set. A width of `0` (the default) draws no visible line.
    """
    if isinstance(config, dict):
        config = ZeroLineConfig(**config)
    elif config is None:
        config = ZeroLineConfig()

    if isinstance(border, dict):
        border = BorderConfig(**border)
    elif border is None:
        border = BorderConfig()

    color = config.color if config.color is not None else border.color
    rule = alt.Chart().mark_rule(color=to_altair_optional(color), strokeWidth=config.width)

    if vertical:
        return rule.encode(x=alt.datum(0))
    else:
        return rule.encode(y=alt.datum(0))


def shift(
    df: pl.DataFrame | pl.LazyFrame,
    score_col: str,
    rank_col: str,
    score_type_col: str,
    token_col: str,
    max_rank: int = 50,
    vertical: bool = True,
    height: float | None = None,
    width: float | None = None,
    autoscale_axes: bool = True,
    rank_axis: RankAxisConfig | dict | None = None,
    score_axis: InternalScoreAxisConfig | dict | None = None,
    bar_config: InternalBarConfig | None = None,
    label_config: LabelConfig | dict | None = None,
    border_config: BorderConfig | dict | None = None,
    zero_line_config: ZeroLineConfig | dict | None = None,
    scale_factor: float = 1.0,
    show_legend: bool = False,
) -> alt.LayerChart:
    """
    Makes a word shift plot of the top contributing tokens distinguishing two corpora.

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
        token_col:
            Column in the dataframe indicating the name of the token to which the scores correspond
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
            Configuration for the rank axis
        score_axis:
            Configuration for the score axis
        bar_config:
            Configuration for the bars
        label_config:
            Configuration for the bar labels
        border_config:
            Configuration for the axes and border
        zero_line_config:
            Configuration for the zero line at the score-axis origin
        scale_factor:
            Multiplier applied to the scale axis when `autoscale_axes` is `True`. Increasing it
            provides more space for the bar labels
        show_legend:
            Whether to show the chart legend

    Returns:
        An Altair chart of the word shift plot
    """
    df = df.filter(pl.col(rank_col) <= max_rank)
    if isinstance(df, pl.LazyFrame):
        df = df.select(token_col, rank_col, score_type_col, score_col).collect()

    size_param = dict()
    if vertical:
        if width is None:
            size_param["width"] = DEFAULT_SCORE_AXIS_PIXEL_SIZE
        else:
            size_param["width"] = width
    else:
        if height is None:
            size_param["height"] = DEFAULT_SCORE_AXIS_PIXEL_SIZE
        else:
            size_param["height"] = height

    bars = shift_bars(
        df,
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        max_rank=max_rank,
        filter_max_rank=False,
        vertical=vertical,
        rank_axis=rank_axis,
        score_axis=score_axis,
        bars=bar_config,
        border=border_config,
        show_legend=show_legend,
        **size_param,
    )

    labels = shift_labels(
        df,
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        max_rank=max_rank,
        filter_max_rank=False,
        vertical=vertical,
        config=label_config,
    )

    chart_layers: list[alt.Chart | alt.LayerChart] = [bars, labels]

    if autoscale_axes:
        scales = shift_scales(
            df,
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            max_rank=max_rank,
            filter_max_rank=False,
            vertical=vertical,
            config=label_config,
            scale_factor=scale_factor,
            **size_param,
        )

        chart_layers.append(scales)

    zero_rule = zero_line(zero_line_config, border_config, vertical=vertical)
    chart_layers.append(zero_rule)

    return cast(alt.LayerChart, alt.layer(*chart_layers))
