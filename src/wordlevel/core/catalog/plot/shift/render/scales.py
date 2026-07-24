from typing import cast

import altair as alt
import polars as pl

from polars import col

from wordlevel.configs import DEFAULT_LABEL_SIZE, LabelConfig
from wordlevel.core.catalog.plot.shift.render.labels import get_label_positions

# Minimum fraction of the score axis reserved for a bar when its label is too wide to fit. Floors
# the available bar pixels so a pathologically long label overflows instead of blowing up the domain
MIN_BAR_PX_FRACTION = 0.1


def shift_scales(
    df: pl.DataFrame,
    score_col: str,
    rank_col: str,
    token_col: str,
    max_rank: int = 50,
    filter_max_rank: bool = True,
    vertical: bool = True,
    width: float | None = None,
    height: float | None = None,
    config: LabelConfig | dict | None = None,
    scale_factor: float = 1.0,
) -> alt.LayerChart:
    """
    Creates a (mostly invisible) Altair chart that adjusts the score axis scale of a shift plot such
    that the limits accomodate bars, labels, and padding, such that they fit within the borders/axes
    of the plot.

    Assumes a dataframe which has one row per word per type of contribution
    """
    if isinstance(config, dict):
        config = LabelConfig(**config)
    elif config is None:
        config = LabelConfig()

    if config.size is None:
        fontsize = DEFAULT_LABEL_SIZE
    else:
        fontsize = config.size

    if vertical:
        if width is None:
            raise ValueError("To scale axis to labels, must specify the chart's width")
        size = width
    else:
        if height is None:
            raise ValueError("To scale axis to labels, must specify the chart's height")
        size = height

    if filter_max_rank:
        df = df.filter(col(rank_col) <= max_rank)

    # Get the base calculations for the scale layers
    df_lims = get_scale_limits(
        df=df,
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        size=size,
        fontsize=fontsize,
        on_bar_ends=config.on_bar_ends,
        pad=config.pad,
        font_width_to_height_ratio=config.font_width_to_height_ratio,
        scale_factor=scale_factor,
    )

    # Create scale layers, one in the positive direction, one in the negative direction
    # tooltip = None disables interactivity, so these invisible points cannot be seen
    base = alt.Chart(df_lims)
    if vertical:
        positive_limit = base.mark_rule(opacity=0).encode(
            x="positive_limit:Q", y=alt.value(0), tooltip=alt.value(None)
        )
        negative_limit = base.mark_rule(opacity=0).encode(
            x="negative_limit:Q", y=alt.value(0), tooltip=alt.value(None)
        )
    else:
        positive_limit = base.mark_rule(opacity=0).encode(
            x=alt.value(0), y="positive_limit:Q", tooltip=alt.value(None)
        )
        negative_limit = base.mark_rule(opacity=0).encode(
            x=alt.value(0), y="negative_limit:Q", tooltip=alt.value(None)
        )

    return cast(alt.LayerChart, alt.layer(positive_limit, negative_limit))


def get_bar_lengths(
    df: pl.DataFrame, score_col: str, rank_col: str, token_col: str
) -> pl.DataFrame:
    """
    Gets each token's positiion, and takes the absolute value to get the magnitude of each token's
    bar
    """
    df_positions = get_label_positions(
        df, score_col=score_col, rank_col=rank_col, token_col=token_col
    )

    return df_positions.with_columns(col(score_col).abs().alias("bar_length"))


def get_scale_limits(
    df: pl.DataFrame,
    score_col: str,
    rank_col: str,
    token_col: str,
    size: float,
    fontsize: float,
    on_bar_ends: bool = True,
    pad: float = 5,
    font_width_to_height_ratio: float = 0.6,
    scale_factor: float = 1.0,
) -> pl.DataFrame:
    """
    Creates a dataframe with one row, which indicates the limits that should be set for the score
    axis to accomodate all bars, labels, and padding. The dataframe has three columns: `rank_col`
    (the position for plotting the limits), "positive_limit", and "negative_limit". If there are
    positive and negative scores, then the limits will be symmetric in the domain [-L, L]. If all
    scores are positive, then the domain will be [0, L]

    This function estimates the limit L that needs to be set.

    To do, note that the user passes the size, in pixels. If all scores are positive, the
    maximum_available_px for bars, labels, and pads is the same as the size. Otherwise,
    maximum_available_px = size / 2.

    A token's bar, label, and padding must fit within the maximum available number of pixels
        b_i_px + l_i_px + 2 * pad <= max_available_px

    Note, b_i_px = (b_i / L) * max_available_px, where b_i is the length of the bar in data units

    Therefore, by substituting for b_i_px and solving for L (as a function of i), we get
        L_i >= (b_i * max_available_px) / (max_available_px - l_i_px - 2 * pad)

    Then, L = max_i L_i

    We also do some edge case handling if l_i_px > max_available_px, and if on_bar_ends=False
    """
    df_bar_lengths = get_bar_lengths(
        df, score_col=score_col, rank_col=rank_col, token_col=token_col
    )

    # The maximum available number of pixels for a bar and its label is all pixels if all scores are
    # positive. Otherwise, a bar can take up half that many pixels
    any_negative_scores = df_bar_lengths.select((col(score_col) < 0).max()).item()
    if any_negative_scores:
        max_available_px = size / 2
    else:
        max_available_px = size

    # Estimated label width = # chars * fontsize * width-to-height-ratio
    # This is because fontsize = height of characters in pixels
    label_width_px_i = (
        col(token_col).str.len_chars().cast(pl.Int64) * font_width_to_height_ratio * fontsize
    ).alias("label_width")

    # A label only extends the domain when it sits at the bar tip. When it sits at the zero line
    # (on_bar_ends=False) it is pixel-fixed at the center, so it does not affect our estimate of the
    # limit L (see below)
    if on_bar_ends:
        label_reservation_px = col("label_width")
    else:
        label_reservation_px = pl.lit(0)

    # A token's bar, label, and padding must fit within the maximum available number of pixels
    #   b_i_px + l_i_px + 2 * pad <= max_available_px
    # available_bar_px_i is the pixels left for the bar after accounting for its label and padding.
    #   available_bar_px_i = max_available_px - l_i_px - 2 * pad
    # A label wider than the axis drives this <= 0 (unsolvable), so floor it: that token keeps a
    # minimum bar and its label overflows the axis
    available_bar_px_i = pl.max_horizontal(
        max_available_px - label_reservation_px - 2 * pad,
        MIN_BAR_PX_FRACTION * max_available_px,
    )
    # Note, b_i_px = (b_i / L) * max_available_px, where b_i is the length of the bar in data units
    # and L is the scale maximum in data units. L is the quantity that we need to determine.
    # Substituting b_i_px and solving for L gives number of pixels need for each token:
    #   L_i >= (b_i * max_available_px) / available_bar_px_i
    total_width_px_i = (col("bar_length") * max_available_px / available_bar_px_i).alias(
        "total_width"
    )

    # We can determine L by taking the largest L_i
    positive_limit = col("total_width").max().alias("positive_limit")

    # If there are any any negative scores, then the domain is [-L, L]. Otherwise, it's [0, L]
    if any_negative_scores:
        negative_limit = (-1 * positive_limit).alias("negative_limit")
    else:
        negative_limit = pl.lit(0).alias("negative_limit")

    # Bring it all together, and add a dummy "rank" column for plotting
    df_lims = (
        df_bar_lengths.with_columns(label_width_px_i)
        .with_columns(total_width_px_i)
        .select(
            pl.lit(1).alias(rank_col),
            positive_limit * scale_factor,
            negative_limit * scale_factor,
        )
    )

    return df_lims
