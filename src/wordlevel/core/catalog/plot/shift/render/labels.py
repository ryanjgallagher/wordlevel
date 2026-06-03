import math

import altair as alt
import polars as pl

from polars import col

from wordlevel.configs import LabelConfig
from wordlevel.core.catalog.plot.utils import to_altair_optional


def shift_labels(
    df: pl.DataFrame,
    score_col: str,
    rank_col: str,
    token_col: str,
    max_rank: int = 50,
    filter_max_rank: bool = True,
    vertical: bool = True,
    config: LabelConfig | dict | None = None,
) -> alt.Chart:
    """
    Assumes a dataframe which has one row per word per type of contribution. There should be columns
    for the contribution, contribution type, and rank
    """
    if filter_max_rank:
        df = df.filter(col(rank_col) <= max_rank)

    df = get_label_positions(df, token_col=token_col, rank_col=rank_col, score_col=score_col)

    # Set default label configurations
    if isinstance(config, dict):
        config = LabelConfig(**config)
    elif config is None:
        config = LabelConfig()

    # Set angle defaults
    if vertical:
        pos_bar_angle = 0
    else:
        pos_bar_angle = 270

    neg_bar_angle = (pos_bar_angle + 180) % 360

    if config.rotation is None:
        rotation = pos_bar_angle
    else:
        rotation = config.rotation

    # Configure text angles, including orienting them so they are readable
    pos_text_angle = to_readable_angle(rotation)
    if config.mirror_rotation:
        neg_text_angle = to_readable_angle(-1 * rotation)
    else:
        neg_text_angle = pos_text_angle

    # Configure the label positions given the bar and text angles
    pos_position_params = get_position_params(
        bar_angle=pos_bar_angle, text_angle=pos_text_angle, pad=config.pad
    )
    neg_position_params = get_position_params(
        bar_angle=neg_bar_angle, text_angle=neg_text_angle, pad=config.pad
    )

    # Configure the axes based on orientation
    if vertical:
        y_axis = alt.Y(f"{rank_col}:O")
        if config.on_bar_ends:
            x_axis = alt.X(score_col)
        else:
            x_axis = alt.datum(0)
    else:
        x_axis = alt.X(f"{rank_col}:O")
        if config.on_bar_ends:
            y_axis = alt.Y(score_col)
        else:
            y_axis = alt.datum(0)

    # Create chart layer
    labels_chart = (
        alt.Chart(df)
        .mark_text(
            color=config.color,
            fontSize=to_altair_optional(config.size),
            fontWeight=config.weight,  # type: ignore[arg-type]
            align=to_conditional_expr(score_col, "align", pos_position_params, neg_position_params),
            baseline=to_conditional_expr(
                score_col, "baseline", pos_position_params, neg_position_params
            ),
            dx=to_conditional_expr(score_col, "dx", pos_position_params, neg_position_params),
            dy=to_conditional_expr(score_col, "dy", pos_position_params, neg_position_params),
        )
        .encode(
            x=x_axis,
            y=y_axis,
            text=token_col,
            angle=alt.condition(
                f"datum['{score_col}'] >= 0", alt.value(pos_text_angle), alt.value(neg_text_angle)
            ),
        )
    )

    return labels_chart


def get_label_positions(
    df: pl.DataFrame, score_col: str, rank_col: str, token_col: str
) -> pl.DataFrame:
    """
    Get positions for labels, assuming long dataframe. Positions are still in score col
    """
    df = (
        df.group_by([token_col, rank_col])
        .agg(
            col(score_col).filter(pl.col(score_col) >= 0).sum().fill_null(0.0).alias("positives"),
            col(score_col).filter(pl.col(score_col) < 0).sum().fill_null(0.0).alias("negatives"),
        )
        .with_columns(
            pl.when(col("positives") > col("negatives").abs())
            .then(col("positives"))
            .otherwise(col("negatives"))
            .alias(score_col)
        )
        .drop("positives", "negatives")
    )

    return df


def to_readable_angle(angle: float) -> float:
    """
    Adjusts an angle to be in a readable orientation
    """
    modulo_angle = angle % 360

    if modulo_angle >= 90 and modulo_angle < 270:
        # Angles in [90, 270) are not properly readable, so they need to be flipped by 180 degrees
        return (angle + 180) % 360
    else:
        # Otherwise, the original angle is readable
        return angle


def get_position_params(bar_angle: float, text_angle: float, pad: float) -> dict[str, str | float]:
    """
    Calculates the static alignment and padding for a single bar direction
    and text rotation.

    This determines where text should sit relative to a specific vector.
    For example, if a bar grows to the right (0 deg) and text is rotated 90 deg,
    the text should be 'left' aligned and 'middle' baselined to sit correctly
    at the tip.

    Args:
        bar_angle: The angle (in degrees) the bar is growing towards.
            (0 = Right, 90 = Down, 180 = Left, 270 = Up).
        rotation: The angle (in degrees) of the text.
        pad: The scalar padding amount (in pixels).

    Returns:
        A dictionary with static values:
            - 'align': 'left', 'right', or 'center'
            - 'baseline': 'top', 'bottom', or 'middle'
            - 'dx': Horizontal offset (float)
            - 'dy': Vertical offset (float)
    """
    relative_angle_rad = math.radians((bar_angle - text_angle) % 360)
    cos_angle = math.cos(relative_angle_rad)
    sin_angle = math.sin(relative_angle_rad)

    # Horizontal alignment
    if abs(cos_angle) < 0.1:
        align = "center"
    elif cos_angle > 0:
        # Bar points right relative to text -> Text starts here
        align = "left"
    else:
        # Bar points left relative to text -> Text ends here
        align = "right"

    # Vertical alignment
    if abs(sin_angle) < 0.1:
        base = "middle"
    elif sin_angle > 0:
        # Bar points down relative to text -> Text is above
        base = "top"
    else:
        # Bar points up relative to text -> Text is below
        base = "bottom"

    # Padding
    dx = cos_angle * pad
    dy = sin_angle * pad

    return {"align": align, "baseline": base, "dx": dx, "dy": dy}


def to_conditional_expr(
    score_col: str,
    key: str,
    positive_params: dict[str, str | float],
    negative_params: dict[str, str | float],
) -> alt.ExprRef:
    """ """
    pos_val = positive_params[key]
    neg_val = negative_params[key]

    if isinstance(pos_val, str):
        pos_str = f"'{pos_val}'"
        neg_str = f"'{neg_val}'"
    else:
        pos_str = str(pos_val)
        neg_str = str(neg_val)

    return alt.expr(f"datum['{score_col}'] >= 0 ? {pos_str} : {neg_str}")
