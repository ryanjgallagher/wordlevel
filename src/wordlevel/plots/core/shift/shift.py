import altair as alt
import polars as pl

from wordlevel.configs import (
    BorderConfig,
    LabelConfig,
    RankAxisConfig,
    TotalsConfig,
    ZeroLineConfig,
)
from wordlevel.constants import DEFAULT_SCORE_AXIS_PIXEL_SIZE
from wordlevel.plots.core.shift.bars import (
    InternalBarConfig,
    InternalScoreAxisConfig,
    shift_bars,
)
from wordlevel.plots.core.shift.border import InternalBorderConfig, shift_borders
from wordlevel.plots.core.shift.labels import shift_labels
from wordlevel.plots.core.shift.scales import shift_scales
from wordlevel.plots.core.shift.totals import shift_total
from wordlevel.plots.core.shift.zero_line import zero_line


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
    show_totals: bool = False,
    rank_axis_config: RankAxisConfig | dict | None = None,
    score_axis_config: InternalScoreAxisConfig | dict | None = None,
    bar_config: InternalBarConfig | None = None,
    label_config: LabelConfig | dict | None = None,
    border_config: BorderConfig | dict | None = None,
    zero_line_config: ZeroLineConfig | dict | None = None,
    totals_config: TotalsConfig | dict | None = None,
    scale_factor: float = 1.0,
    show_legend: bool = False,
) -> alt.LayerChart | alt.VConcatChart | alt.HConcatChart:
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
            `RankAxisConfig.bar_width` to control the height
        width:
            The width of the chart, in pixels. Ignored if horizontal (`not vertical`), see
            `RankAxisConfig.bar_width` to control the width
        show_totals:
            Whether to show the overall total contributions of each score type and overall as an
            additional chart appended to the main shift chart
        rank_axis_config:
            Configuration for the rank axis
        score_axis_config:
            Configuration for the score axis
        bar_config:
            Configuration for the bars
        label_config:
            Configuration for the bar labels
        border_config:
            Configuration for the axes and border
        zero_line_config:
            Configuration for the zero line at the score-axis origin
        totals_config
            Configuration for the "grand" total bar. Ignored if `show_totals` is False
        scale_factor:
            Multiplier applied to the score axis when fitting it to the bar labels. Increasing it
            provides more space for the bar labels
        show_legend:
            Whether to show the chart legend

    Returns:
        An Altair chart of the word shift plot
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

    # Resolve the axis and border configs once so they can be shared with the border framing
    if isinstance(rank_axis_config, dict):
        rank_axis_config = RankAxisConfig(**rank_axis_config)
    elif rank_axis_config is None:
        rank_axis_config = RankAxisConfig()

    if isinstance(score_axis_config, dict):
        score_axis_config = InternalScoreAxisConfig(**score_axis_config)
    elif score_axis_config is None:
        score_axis_config = InternalScoreAxisConfig()

    if isinstance(border_config, dict):
        border_config = BorderConfig(**border_config)
    elif border_config is None:
        border_config = BorderConfig()

    border_is_visible = border_config.width > 0

    # For horizontal totals (`vertical=False` and `show_totals=True`), the main chart's score axis
    # is moved to the far side, so its native score-axis domain draws that outer edge. The
    # grand-total panel draws the seam between them instead
    score_axis_on_far_side = show_totals and not vertical

    # The seams separating the totals panels from the main chart are structural, so they are drawn
    # even when the border box is off. They then take the score axis width, matching the score-axis
    # domain that already separates the two totals panels. Note, `border_is_visible` is derived from
    # the user-facing width above, so this override cannot turn the border box back on
    seam_width = border_config.width if border_is_visible else score_axis_config.width

    main_border = InternalBorderConfig.from_config(
        border_config,
        draw_score_axis=True,
        draw_rank_axis=True,
        # Vertical: the top edge is the seam with the grand-total panel.
        # Horizontal: the score axis has moved, so the grand total panel draws that seam instead
        draw_border_opposite_score_axis=(
            (border_is_visible or show_totals) and not score_axis_on_far_side
        ),
        draw_border_opposite_rank_axis=border_is_visible,
        width=seam_width,
    )

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
        border_config=main_border,
        show_legend=show_legend,
        score_axis_opposite=score_axis_on_far_side,
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

    chart = shift_borders(
        [bars, labels, scales, zero_rule],
        rank_col=rank_col,
        score_col=score_col,
        rank_axis_config=rank_axis_config,
        score_axis_config=score_axis_config,
        border_config=main_border,
        vertical=vertical,
    )

    if show_totals:
        grand_total = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=True,
            vertical=vertical,
            rank_axis_config=rank_axis_config,
            score_axis_config=score_axis_config,
            bar_config=bar_config,
            label_config=label_config,
            border_config=InternalBorderConfig.from_config(
                border_config,
                draw_score_axis=False,
                draw_rank_axis=True,
                # Vertical: the seam with the per-type totals. Horizontal: the seam with the main
                # chart
                draw_border_opposite_score_axis=True,
                draw_border_opposite_rank_axis=border_is_visible,
                width=seam_width,
            ),
            totals_config=totals_config,
            zero_line_config=zero_line_config,
            scale_factor=scale_factor,
            **size_param,
        )
        shift_type_totals = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=False,
            vertical=vertical,
            rank_axis_config=rank_axis_config,
            score_axis_config=score_axis_config,
            bar_config=bar_config,
            label_config=label_config,
            border_config=InternalBorderConfig.from_config(
                border_config,
                # Vertical: falls on the seam that the grand-total panel draws. Horizontal: an
                # outer edge
                draw_score_axis=border_is_visible and score_axis_on_far_side,
                draw_rank_axis=True,
                # Vertical: an outer edge. Horizontal: the seam with the grand-total panel
                draw_border_opposite_score_axis=border_is_visible or score_axis_on_far_side,
                draw_border_opposite_rank_axis=border_is_visible,
                width=seam_width,
            ),
            totals_config=totals_config,
            zero_line_config=zero_line_config,
            scale_factor=scale_factor,
            **size_param,
        )

        if vertical:
            alt_concat = alt.vconcat
        else:
            alt_concat = alt.hconcat

        chart = alt_concat(
            shift_type_totals,
            grand_total,
            chart,
            spacing=0,
            bounds="flush",
        ).resolve_scale(color="independent", fillOpacity="independent")

    return chart
