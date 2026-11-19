import altair as alt
import polars as pl

from wordlevel.configs import (
    BorderConfig,
    LabelConfig,
    RankAxisConfig,
    TotalsConfig,
    ZeroLineConfig,
)
from wordlevel.plots.core.shift.bars import InternalBarConfig, InternalScoreAxisConfig
from wordlevel.plots.core.shift.border import InternalBorderConfig
from wordlevel.plots.core.shift.shift import shift
from wordlevel.plots.core.shift.totals import shift_total


def compose_shift(
    df: pl.DataFrame | pl.LazyFrame,
    score_col: str,
    rank_col: str,
    score_type_col: str,
    token_col: str,
    rank_axis_config: RankAxisConfig,
    score_axis_config: InternalScoreAxisConfig,
    bar_config: InternalBarConfig,
    label_config: LabelConfig,
    border_config: BorderConfig,
    zero_line_config: ZeroLineConfig,
    totals_config: TotalsConfig,
    max_rank: int = 50,
    vertical: bool = True,
    height: float | None = None,
    width: float | None = None,
    show_totals: bool = False,
    scale_factor: float = 1.0,
    show_legend: bool = False,
) -> alt.LayerChart | alt.VConcatChart | alt.HConcatChart:
    """
    Makes a word shift plot of the top contributing tokens distinguishing two corpora, optionally
    composed with panels of the total contributions.

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
            Configuration for the axes and border
        zero_line_config:
            Configuration for the zero line at the score-axis origin
        totals_config:
            Configuration for the totals panels. Ignored if `show_totals` is False
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
            Whether to show the overall total contributions of each score type and overall as
            additional panels composed with the main shift panel
        scale_factor:
            Multiplier applied to the score axis when fitting it to the bar labels. Increasing it
            provides more space for the bar labels
        show_legend:
            Whether to show the chart legend

    Returns:
        An Altair chart of the word shift plot
    """
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

    chart = shift(
        df,
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        token_col=token_col,
        max_rank=max_rank,
        vertical=vertical,
        height=height,
        width=width,
        rank_axis_config=rank_axis_config,
        score_axis_config=score_axis_config,
        bar_config=bar_config,
        label_config=label_config,
        border_config=main_border,
        zero_line_config=zero_line_config,
        scale_factor=scale_factor,
        show_legend=show_legend,
        score_axis_opposite=score_axis_on_far_side,
    )

    if not show_totals:
        return chart

    grand_total = shift_total(
        df,
        score_col=score_col,
        score_type_col=score_type_col,
        grand_total=True,
        vertical=vertical,
        height=height,
        width=width,
        rank_axis_config=rank_axis_config,
        score_axis_config=score_axis_config,
        bar_config=bar_config,
        label_config=label_config,
        border_config=InternalBorderConfig.from_config(
            border_config,
            draw_score_axis=False,
            draw_rank_axis=True,
            # Vertical: the seam with the per-type totals. Horizontal: the seam with the main chart
            draw_border_opposite_score_axis=True,
            draw_border_opposite_rank_axis=border_is_visible,
            width=seam_width,
        ),
        totals_config=totals_config,
        zero_line_config=zero_line_config,
        scale_factor=scale_factor,
    )
    shift_type_totals = shift_total(
        df,
        score_col=score_col,
        score_type_col=score_type_col,
        grand_total=False,
        vertical=vertical,
        height=height,
        width=width,
        rank_axis_config=rank_axis_config,
        score_axis_config=score_axis_config,
        bar_config=bar_config,
        label_config=label_config,
        border_config=InternalBorderConfig.from_config(
            border_config,
            # Vertical: falls on the seam that the grand-total panel draws. Horizontal: an outer
            # edge
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
    )

    if vertical:
        alt_concat = alt.vconcat
    else:
        alt_concat = alt.hconcat

    return alt_concat(
        shift_type_totals,
        grand_total,
        chart,
        spacing=0,
        bounds="flush",
    ).resolve_scale(color="independent", fillOpacity="independent")
