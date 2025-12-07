import altair as alt
import polars as pl

from wordlevel.plots.shift.bars import shift_bars, RankAxisConfig, ScoreAxisConfig, BarConfig
from wordlevel.plots.shift.labels import shift_labels, LabelConfig
from wordlevel.plots.shift.scales import shift_scales


def shift(
    df: pl.DataFrame,
    score_col: str,
    rank_col: str,
    score_type_col: str,
    label_col: str,
    max_rank: int = 50,
    vertical: bool = True,
    height: float | None = None,
    width: float | None = None,
    autoscale_axes: bool = True,
    rank_axis: RankAxisConfig | dict | None = None,
    score_axis: ScoreAxisConfig | dict | None = None,
    bar_config: BarConfig | dict | None = None,
    label_config: LabelConfig | dict | None = None,
    scale_factor: float = 1.0,
    show_legend: bool = False,
    show_border: bool = False,
) -> alt.LayerChart:
    """ """
    df = df.filter(pl.col(rank_col) <= max_rank)

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
        label_col=label_col,
        max_rank=max_rank,
        filter_max_rank=False,
        vertical=vertical,
        config=label_config,
    )

    chart_layers = [bars, labels]

    if autoscale_axes:
        scales = shift_scales(
            df,
            score_col=score_col,
            rank_col=rank_col,
            label_col=label_col,
            max_rank=max_rank,
            filter_max_rank=False,
            vertical=vertical,
            width=width,
            height=height,
            config=label_config,
            scale_factor=scale_factor,
        )

        chart_layers.append(scales)

    return alt.layer(*chart_layers)
