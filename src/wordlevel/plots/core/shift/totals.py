from dataclasses import asdict, replace
from typing import cast

import altair as alt
import polars as pl
from polars import col

from wordlevel.configs import (
    LABEL_SUFFIX,
    LabelConfig,
    RankAxisConfig,
    TotalsConfig,
    ZeroLineConfig,
)
from wordlevel.constants import (
    COUNTERACTED_SUFFIX,
    DEFAULT_SCORE_AXIS_PIXEL_SIZE,
    GRAND_TOTAL_SCORE_TYPE,
)
from wordlevel.plots.core.shift.bars import (
    InternalBarConfig,
    InternalScoreAxisConfig,
    shift_bars,
)
from wordlevel.plots.core.shift.border import (
    _TRANSLATE_PX,
    InternalBorderConfig,
    shift_borders,
)
from wordlevel.plots.core.shift.labels import shift_labels
from wordlevel.plots.core.shift.scales import shift_scales
from wordlevel.plots.core.shift.zero_line import zero_line
from wordlevel.plots.core.utils import to_altair_optional


_SCORE_TYPE_COL = "score_type"
_POSITION_COL = "position"  # analogous to `rank_col`
_LABEL_COL = "label"  # analogous to `token_col`
_ALIGNMENT_COL = "alignment"

_SHIFT_TYPE_SUMMARY_POSITIONS: dict[str, int] = {
    "more_of_positive": 1,
    "less_of_positive": 1,
    "more_of_negative": 2,
    "less_of_negative": 2,
    "higher_score": 3,
    "lower_score": 3,
}

_SHIFT_SUMMARY_DROP_IF_EMPTY: frozenset[str] = frozenset({"higher_score", "lower_score"})


def _get_score_type_labels(config: TotalsConfig) -> dict[str, str]:
    """
    Resolves a `TotalsConfig` into its score type to label mapping. The config carries a label for
    every contribution type of either plot, so the labels that do not name a score type present in
    the data simply never match. The grand total is labeled separately, since it is a sum across
    contribution types rather than one of them
    """
    return {
        name.removesuffix(LABEL_SUFFIX): label
        for name, label in asdict(config).items()
        if name.endswith(LABEL_SUFFIX) and name != "grand_total_label"
    }


def _get_total(
    df: pl.DataFrame | pl.LazyFrame, score_col: str, label: str
) -> pl.DataFrame | pl.LazyFrame:
    """
    Takes a long dataframe of one row per token per score type and transforms it into a dataframe
    with a single row representing the total across all tokens and contributions. The `label` is the
    annotation plotted next to the bar
    """
    return df.select(
        pl.lit(GRAND_TOTAL_SCORE_TYPE).alias(_SCORE_TYPE_COL),
        pl.lit(1).alias(_POSITION_COL),
        pl.lit(label).alias(_LABEL_COL),
        col(score_col).sum(),
    )


def _get_score_type_totals(
    df: pl.DataFrame | pl.LazyFrame,
    score_col: str,
    score_type_col: str,
    labels: dict[str, str],
) -> pl.DataFrame | pl.LazyFrame:
    """
    Takes a long dataframe of one row per token per score type and transforms it into a dataframe
    with one row per score type and its total across all tokens. The `labels` are the annotations
    plotted next to each type of score type bar. Assumes this mapping matches the score types that
    appear in `score_type_col`
    """
    # score_type -> label
    label_expr = pl.when(False).then(pl.lit(""))
    for score_type, label in labels.items():
        label_expr = label_expr.when(col(score_type_col) == score_type).then(pl.lit(label))

    # score_type -> position
    pos_expr = pl.when(False).then(pl.lit(0))
    for score_type, pos in _SHIFT_TYPE_SUMMARY_POSITIONS.items():
        pos_expr = pos_expr.when(col(score_type_col) == score_type).then(pl.lit(pos))
    # All other score types default to just being on a single row in the totals
    pos_expr = pos_expr.otherwise(pl.lit(1))

    # score_type -> if it should be dropped if both components are null (higher/lower score)
    drop_score_type_if_empty = col(_SCORE_TYPE_COL).is_in(_SHIFT_SUMMARY_DROP_IF_EMPTY)

    # 1. Get sums for each score type. "Counteracted" scores still contribute to that score type
    df_totals = (
        df.select(col(score_type_col).str.strip_suffix(COUNTERACTED_SUFFIX), score_col)
        .group_by(score_type_col)
        .agg(col(score_col).sum())
        .with_columns(
            pos_expr.alias(_POSITION_COL),
            label_expr.alias(_LABEL_COL),
        )
        .rename({score_type_col: _SCORE_TYPE_COL})
    )
    # 2. Drop higher/lower scores if they're both 0 (i.e. there are no score differences)
    df_totals = (
        df_totals.with_columns(
            pl.when(drop_score_type_if_empty)
            .then(col(score_col).abs())
            .otherwise(0.0)  # Other score types don't contribute to this sum
            .sum()  # If both are 0, then this sum will be 0
            .alias("score_for_dropping")
        )
        .filter(~drop_score_type_if_empty | (col("score_for_dropping") > 0))
        .drop("score_for_dropping")
    )
    # 3. Make sure that labels don't overlap in the case that a component sums to zero
    df_totals = df_totals.sort(_POSITION_COL, _LABEL_COL).with_columns(
        # If the bar's sign isn't zero, then point the label in that direction
        pl.when(col(score_col).sign() != 0)
        .then(col(score_col).sign())
        # If the bar's sign is zero, but the other bar's sign is not, then take the opposite
        # sign of the nonzero bar. Note, we group "pairs" of bars by `POSITION_COL`
        .when(col(score_col).sign().sum().over(_POSITION_COL) != 0)
        .then(-1 * col(score_col).sign().sum().over(_POSITION_COL))
        # Otherwise, place the first label we see at the positive position, other at negative
        .otherwise(
            pl.when(col(_LABEL_COL).cum_count().over(_POSITION_COL) == 1).then(1.0).otherwise(-1.0)
        )
        .alias(_ALIGNMENT_COL)
    )

    return df_totals


def shift_total(
    df: pl.DataFrame | pl.LazyFrame,
    score_col: str,
    score_type_col: str,
    grand_total: bool,
    vertical: bool = True,
    height: float | None = None,
    width: float | None = None,
    rank_axis_config: RankAxisConfig | dict | None = None,
    score_axis_config: InternalScoreAxisConfig | dict | None = None,
    bar_config: InternalBarConfig | None = None,
    label_config: LabelConfig | dict | None = None,
    border_config: InternalBorderConfig | None = None,
    totals_config: TotalsConfig | dict | None = None,
    zero_line_config: ZeroLineConfig | dict | None = None,
    scale_factor: float = 1.0,
) -> alt.LayerChart:
    """
    Makes a bar chart representing word shift contribution totals, which can either be per score
    type or overall (the "grand total").

    Args:
        df:
            A long dataframe with one row per token per type of score to plot. For example, if a
            token has two contributing components `more_of_positive` and `higher_score`, then it
            should have two rows, where the `score_col` indicates the length of each
        score_col:
            Column in the dataframe indicating the "score" associated with each bar, i.e. its length
        score_type_col:
            Column in the dataframe indicating the type of "score" associated with each bar
        grand_total:
            Whether to create the bar chart for individual score types or just the grand total (all
            contributions summed together)
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
            Configuration for the bars, controls the colors used to plot per score type totals
        label_config:
            Configuration for the bar labels
        border_config:
            Configuration for the axes and border
        totals_config:
            Configuration for the totals panels, controls the label plotted next to each bar and
            the color of the grand total bar
        zero_line_config:
            Configuration for the zero line at the score-axis origin
        scale_factor:
            Multiplier applied to the score axis when fitting it to the bar labels. Increasing it
            provides more space for the bar labels

    Returns:
        An Altair chart of word shift total(s)
    """
    # Initialize necessary configs
    if isinstance(rank_axis_config, dict):
        rank_axis_config = RankAxisConfig(**rank_axis_config)
    elif rank_axis_config is None:
        rank_axis_config = RankAxisConfig()

    if isinstance(score_axis_config, dict):
        score_axis_config = InternalScoreAxisConfig(**score_axis_config)
    elif score_axis_config is None:
        score_axis_config = InternalScoreAxisConfig()

    if isinstance(totals_config, dict):
        totals_config = TotalsConfig(**totals_config)
    elif totals_config is None:
        totals_config = TotalsConfig()

    # The score type and grand total configs need to be resolved separately since they're both used
    if isinstance(bar_config, dict):
        type_bar_config = InternalBarConfig(**bar_config)
    elif bar_config is None:
        # TODO: these raw score types keep the `_counteracted` suffix that `_get_score_type_totals`
        #       strips, so the fallback labels/colors won't match; strip before `from_score_types`
        score_types = df.lazy().select(col(score_type_col).unique()).collect().to_series().to_list()
        type_bar_config = InternalBarConfig.from_score_types(score_types)
    else:
        type_bar_config = replace(bar_config, opacities=None)

    if grand_total:
        # The grand total inherits the bar outline unless the `TotalsConfig` sets its own
        bar_config = replace(
            type_bar_config,
            colors={GRAND_TOTAL_SCORE_TYPE: type_bar_config.colors[GRAND_TOTAL_SCORE_TYPE]},
            orders=None,
            outline_color=(
                totals_config.grand_total_outline_color
                if totals_config.grand_total_outline_color is not None
                else type_bar_config.outline_color
            ),
            outline_width=(
                totals_config.grand_total_outline_width
                if totals_config.grand_total_outline_width is not None
                else type_bar_config.outline_width
            ),
        )
    else:
        bar_config = type_bar_config

    if border_config is None:
        border_config = InternalBorderConfig()

    # Size parameters
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

    # Get the various totals. Note, both are needed to make sure they use a common scale
    df_grand_total, df_score_type_totals = pl.collect_all(
        [
            _get_total(
                df=df,
                score_col=score_col,
                label=totals_config.grand_total_label,
            ).lazy(),
            _get_score_type_totals(
                df=df,
                score_col=score_col,
                score_type_col=score_type_col,
                labels=_get_score_type_labels(totals_config),
            ).lazy(),
        ]
    )

    df_all_totals = pl.concat(
        [
            df_grand_total.select(score_col, _POSITION_COL, _LABEL_COL),
            df_score_type_totals.select(score_col, _POSITION_COL, _LABEL_COL),
        ]
    )

    if grand_total:
        df_total = df_grand_total
    else:
        df_total = df_score_type_totals

    # Get plot components
    bars = shift_bars(
        df=df_total,
        score_col=score_col,
        rank_col=_POSITION_COL,
        score_type_col=_SCORE_TYPE_COL,
        filter_max_rank=False,
        vertical=vertical,
        rank_axis_config=rank_axis_config,
        score_axis_config=score_axis_config,
        bar_config=bar_config,
        border_config=border_config,
        show_legend=False,
        **size_param,
    )
    labels = shift_labels(
        df=df_total,
        score_col=score_col,
        rank_col=_POSITION_COL,
        token_col=_LABEL_COL,
        filter_max_rank=False,
        vertical=vertical,
        label_config=label_config,
        alignment_col=None if grand_total else _ALIGNMENT_COL,
    )
    scales = shift_scales(
        df=df_all_totals,  # Need to use the common scale across all totals and the grand total
        score_col=score_col,
        rank_col=_POSITION_COL,
        token_col=_LABEL_COL,
        filter_max_rank=False,
        vertical=vertical,
        label_config=label_config,
        scale_factor=scale_factor,
        **size_param,
    )
    zero_rule = zero_line(
        zero_line_config=zero_line_config, border_config=border_config, vertical=vertical
    )

    # Modify the returned bar axes to hide information (labels, ticks, etc.)
    if vertical:
        score_channel, rank_channel = bars.encoding.x, bars.encoding.y
    else:
        score_channel, rank_channel = bars.encoding.y, bars.encoding.x

    modified_score_axis = score_channel.axis(
        domain=border_config.draw_score_axis,
        domainWidth=score_axis_config.width,
        domainColor=to_altair_optional(
            score_axis_config.axis_color
            if score_axis_config.axis_color is not None
            else border_config.color
        ),
        title=None,
        labels=False,
        ticks=False,
        grid=False,
        translate=_TRANSLATE_PX,
    ).scale(nice=False)

    modified_rank_axis = rank_channel.axis(
        domain=border_config.draw_rank_axis,
        domainWidth=rank_axis_config.width,
        domainColor=to_altair_optional(
            rank_axis_config.axis_color
            if rank_axis_config.axis_color is not None
            else border_config.color
        ),
        title=None
        if rank_axis_config.title is None
        else "l",  # Hidden, but needed to align axes with main chart
        titleColor="transparent",
        titleFontSize=to_altair_optional(rank_axis_config.title_fontsize),
        ticks=False,
        labelExpr="format(datum.value, '02d')",  # Hidden, but needed to align axes with main chart
        labelColor="transparent",
        grid=False,
        translate=_TRANSLATE_PX,
    )

    if vertical:
        modified_bars = bars.encode(x=modified_score_axis, y=modified_rank_axis)
    else:
        modified_bars = bars.encode(x=modified_rank_axis, y=modified_score_axis)

    chart = shift_borders(
        [modified_bars, labels, scales, zero_rule],
        border_config=border_config,
        rank_col=_POSITION_COL,
        score_col=score_col,
        rank_axis_config=rank_axis_config,
        score_axis_config=score_axis_config,
        vertical=vertical,
    )

    return cast(alt.LayerChart, chart)
