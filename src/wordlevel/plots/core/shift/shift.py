import altair as alt
import polars as pl

from wordlevel.configs import LabelConfig, RankAxisConfig, ZeroLineConfig
from wordlevel.constants import DEFAULT_SCORE_AXIS_PIXEL_SIZE
from wordlevel.plots.core.shift.bars import (
    InternalBarConfig,
    InternalScoreAxisConfig,
    shift_bars,
)
from wordlevel.plots.core.shift.border import InternalBorderConfig, shift_borders
from wordlevel.plots.core.shift.labels import shift_labels
from wordlevel.plots.core.shift.scales import shift_scales
from wordlevel.plots.core.shift.zero_line import zero_line


def shift(
    df: pl.DataFrame | pl.LazyFrame,
    score_col: str,
    rank_col: str,
    score_type_col: str,
    token_col: str,
    rank_axis_config: RankAxisConfig,
    score_axis_config: InternalScoreAxisConfig,
    bar_config: InternalBarConfig,
    label_config: LabelConfig,
    border_config: InternalBorderConfig,
    zero_line_config: ZeroLineConfig,
    max_rank: int = 50,
    vertical: bool = True,
    height: float | None = None,
    width: float | None = None,
    scale_factor: float = 1.0,
    show_legend: bool = False,
    score_axis_opposite: bool = False,
) -> alt.LayerChart:
    """
    Makes a single word shift panel of the top contributing tokens distinguishing two corpora.

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
        rank_axis_config:
            Configuration for the rank axis
        score_axis_config:
            Configuration for the score axis
        bar_config:
            Configuration for the bars
        label_config:
            Configuration for the bar labels
        border_config:
            Configuration for the axes and border, including which of the panel's edges are drawn
        zero_line_config:
            Configuration for the zero line at the score-axis origin
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
        scale_factor:
            Multiplier applied to the score axis when fitting it to the bar labels. Increasing it
            provides more space for the bar labels
        show_legend:
            Whether to show the chart legend
        score_axis_opposite:
            Whether to place the score axis on the side opposite its default position. Ignored if
            `vertical` is True

    Returns:
        An Altair chart of the word shift panel
    """
    df_filt = (
        df.lazy()
        .filter(pl.col(rank_col) <= max_rank)
        .select(token_col, rank_col, score_type_col, score_col)
        .collect()
    )

    size_param = {}
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
        df_filt,
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        max_rank=max_rank,
        filter_max_rank=False,
        vertical=vertical,
        rank_axis_config=rank_axis_config,
        score_axis_config=score_axis_config,
        bar_config=bar_config,
        border_config=border_config,
        show_legend=show_legend,
        score_axis_opposite=score_axis_opposite,
        **size_param,
    )
    labels = shift_labels(
        df_filt,
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        max_rank=max_rank,
        filter_max_rank=False,
        vertical=vertical,
        label_config=label_config,
    )
    scales = shift_scales(
        df_filt,
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        max_rank=max_rank,
        filter_max_rank=False,
        vertical=vertical,
        label_config=label_config,
        scale_factor=scale_factor,
        **size_param,
    )
    zero_rule = zero_line(zero_line_config, border_config, vertical=vertical)

    return shift_borders(
        [bars, labels, scales, zero_rule],
        rank_col=rank_col,
        score_col=score_col,
        rank_axis_config=rank_axis_config,
        score_axis_config=score_axis_config,
        border_config=border_config,
        vertical=vertical,
    )
