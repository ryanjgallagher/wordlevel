import altair as alt
import polars as pl

from polars import col

from wordlevel.plots.shift.labels import LabelConfig, get_label_positions


def shift_scales(
    df: pl.DataFrame,
    score_col: str,
    rank_col: str,
    label_col: str,
    max_rank: int = 50,
    filter_max_rank: bool = True,
    vertical: bool = True,
    width: float | None = None,
    height: float | None = None,
    config: LabelConfig | dict | None = None,
    scale_factor: float = 1.0,
) -> alt.LayerChart:
    """
    Assumes a dataframe which has one row per word per type of contribution. There should be columns
    for the contribution, contribution type, and rank
    """
    if isinstance(config, dict):
        config = LabelConfig(**config)
    elif config is None:
        config = LabelConfig()

    if config.size is None:
        fontsize = LabelConfig().size
    else:
        fontsize = config.size

    if vertical:
        size = width
        if width is None:
            raise ValueError("To scale axis to labels, must specify the chart's width")
    else:
        size = height
        if height is None:
            raise ValueError("To scale axis to labels, must specify the chart's height")

    if filter_max_rank:
        df = df.filter(col(rank_col) <= max_rank)

    # Get the base calculations for the scale layers
    df_lims = get_scale_limits(
        df=df,
        score_col=score_col,
        rank_col=rank_col,
        label_col=label_col,
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

    return alt.layer(positive_limit, negative_limit)


def get_bar_lengths(
    df: pl.DataFrame, score_col: str, rank_col: str, label_col: str
) -> pl.DataFrame:
    """ """
    df_positions = get_label_positions(
        df, score_col=score_col, rank_col=rank_col, label_col=label_col
    )

    return df_positions.select(label_col, rank_col, col(score_col).abs().alias("bar_length"))


def get_scale_limits(
    df: pl.DataFrame,
    score_col: str,
    rank_col: str,
    label_col: str,
    size: float,
    fontsize: float,
    on_bar_ends: bool = True,
    pad: float = 5,
    font_width_to_height_ratio: float = 0.6,
    scale_factor: float = 1.0,
):
    """ """
    df_bar_lengths = get_bar_lengths(
        df, score_col=score_col, rank_col=rank_col, label_col=label_col
    )

    # Get how long the entire score axis is
    len_score_axis = df_bar_lengths.select(col("bar_length").max() * 2).item()
    # Get how many units along the score axis per a single pixel
    units_per_pixel = len_score_axis / size
    # Estimate the average number of pixels wide each character is
    avg_char_size_px = fontsize * font_width_to_height_ratio
    # Estimate the number of units covered by the average character
    label_scaling_factor = avg_char_size_px * units_per_pixel

    # With the scaling information, make the limits symmetric and long enough to account for labels
    df_lims = df_bar_lengths.with_columns(
        (col(label_col).str.len_chars() * label_scaling_factor).alias("label_width")
    )

    if on_bar_ends:
        estimated_width = col("bar_length") + col("label_width")
    else:
        estimated_width = pl.max_horizontal(col("bar_length"), col("label_width"))

    df_lims = df_lims.with_columns(
        (estimated_width + units_per_pixel * pad).alias("estimated_width")
    ).select(
        pl.lit(1).alias(rank_col),
        (scale_factor * col("estimated_width").max()).alias("positive_limit"),
        (-1 * scale_factor * col("estimated_width").max()).alias("negative_limit"),
    )

    return df_lims
