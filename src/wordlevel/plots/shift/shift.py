from typing import cast

import altair as alt
import polars as pl

from wordlevel.plots.shift.bars import (
    shift_bars,
    RankAxisConfig,
    InternalScoreAxisConfig,
    InternalBarConfig,
)
from wordlevel.plots.shift.labels import shift_labels, LabelConfig
from wordlevel.plots.shift.scales import shift_scales


DEFAULT_PIXELS_PER_RANK = 14
DEFAULT_SCORE_AXIS_PIXEL_SIZE = 400


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
    bar_config: InternalBarConfig | dict | None = None,
    label_config: LabelConfig | dict | None = None,
    scale_factor: float = 1.0,
    show_legend: bool = False,
    show_border: bool = False,
) -> alt.LayerChart:
    """ """
    df = df.filter(pl.col(rank_col) <= max_rank)
    if isinstance(df, pl.LazyFrame):
        df = df.select(token_col, rank_col, score_type_col, score_col).collect()

    if vertical:
        if width is None:
            width = DEFAULT_SCORE_AXIS_PIXEL_SIZE
        if height is None:
            height = max(max_rank * DEFAULT_PIXELS_PER_RANK, DEFAULT_SCORE_AXIS_PIXEL_SIZE)
    else:
        if height is None:
            height = DEFAULT_SCORE_AXIS_PIXEL_SIZE
        if width is None:
            width = max(max_rank * DEFAULT_PIXELS_PER_RANK, DEFAULT_SCORE_AXIS_PIXEL_SIZE)

    bars = shift_bars(
        df,
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        max_rank=max_rank,
        filter_max_rank=False,
        vertical=vertical,
        height=height,
        width=width,
        rank_axis=rank_axis,
        score_axis=score_axis,
        bars=bar_config,
        show_legend=show_legend,
        show_border=show_border,
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
            width=width,
            height=height,
            config=label_config,
            scale_factor=scale_factor,
        )

        chart_layers.append(scales)

    return cast(alt.LayerChart, alt.layer(*chart_layers))
