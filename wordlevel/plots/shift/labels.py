import math
from dataclasses import dataclass

import altair as alt
import polars as pl

from polars import col

from wordlevel.plots.utils import to_altair_optional


@dataclass
class LabelConfig:
    """ """

    pad: float = 5
    color: str = "black"
    size: float | None = None
    weight: str | int = "normal"
    rotation: float | None = None
    mirror_rotation: bool = True
    on_bar_ends: bool = True


def shift_labels(
    df: pl.DataFrame,
    score_col: str,
    rank_col: str,
    label_col: str,
    max_rank: int = 50,
    vertical: bool = True,
    config: LabelConfig | None = None,
) -> alt.Chart:
    """
    Assumes a dataframe which has one row per word per type of contribution. There should be columns
    for the contribution, contribution type, and rank
    """
    df = get_label_positions(
        df, label_col=label_col, rank_col=rank_col, score_col=score_col, max_rank=max_rank
    )

    # Set default label configurations
    if config is None:
        config = LabelConfig()

    # Configure the orientation
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

    # Get rotation defaults
    base_angle = 0
    if config.rotation is not None:
        base_angle = config.rotation
    elif not vertical:
        base_angle = 270
    # Transform the angle so that the text is never upside down
    positive_angle = to_readable_angle(base_angle)
    if config.mirror_rotation:
        negative_angle = to_readable_angle(-base_angle)
    else:
        negative_angle = positive_angle
    # Configure the angle encoding based on the rotation
    if positive_angle == negative_angle:
        angle_encoding = alt.value(positive_angle)
    else:
        angle_encoding = alt.condition(
            f"datum['{score_col}'] >= 0", alt.value(positive_angle), alt.value(negative_angle)
        )

    # Configure details for the label positions and padding
    position_params = get_position_params(
        score_col=score_col,
        positive_angle=positive_angle,
        negative_angle=negative_angle,
        pad=config.pad,
        vertical=vertical,
    )

    # Create chart layer
    labels_chart = (
        alt.Chart(df)
        .mark_text(
            color=config.color,
            fontSize=to_altair_optional(config.size),
            fontWeight=config.weight,
            **position_params,
        )
        .encode(x=x_axis, y=y_axis, text=label_col, angle=angle_encoding)
    )

    return labels_chart


def get_label_positions(
    df: pl.DataFrame, score_col: str, rank_col: str, label_col: str, max_rank: int = 50
) -> pl.DataFrame:
    """
    Get positions for labels, assuming long dataframe. Positions are still in score col
    """
    df = (
        df.filter(pl.col(rank_col) <= max_rank)
        .group_by([label_col, rank_col])
        .agg(
            col(score_col).filter(pl.col(score_col) >= 0).sum().fill_null(0).alias("positives"),
            col(score_col).filter(pl.col(score_col) < 0).sum().fill_null(0).alias("negatives"),
        )
        .with_columns(
            pl.when(pl.col("positives") > pl.col("negatives").abs())
            .then(pl.col("positives"))
            .otherwise(pl.col("negatives"))
            .alias(score_col)
        )
    )

    return df


def to_readable_angle(angle: float) -> float:
    """
    Adjusts an angle to be in a readable orientation, [-90, 90]
    """
    modulo_angle = angle % 360

    if modulo_angle >= 90 and modulo_angle < 270:
        return angle + 180

    return angle


def get_position_params(
    score_col: str, positive_angle: float, negative_angle: float, pad: float, vertical: bool
) -> dict:
    """
    Calculates the conditional Altair expressions for text alignment and padding.

    Because positive and negative bars grow in opposite directions (and may have
    different text rotations), this function calculates the geometry for both
    cases and returns Altair expressions (ternary operators) that apply the
    correct settings dynamically based on the data value.

    Args:
        score_col: The name of the column containing the score (used for the
            Altair condition `datum[score_col] >= 0`).
        positive_angle: The text rotation angle (in degrees) for positive bars.
        negative_angle: The text rotation angle (in degrees) for negative bars.
        pad: The distance (in pixels) to offset the text from the bar end.
        vertical: If True, bars are horizontal (growing left/right).
            If False, bars are vertical (growing up/down).

    Returns:
        A dictionary containing Altair expressions for 'align', 'baseline',
        'dx', and 'dy'.
    """
    # Get the direction that bars go
    if vertical:
        bar_angle = 0
    else:
        bar_angle = 270

    # Calculate geometry for the positive side
    pos_position_params = get_position_params_relative_to_bar(bar_angle, positive_angle, pad)
    # Calculate geometry for the negative side - opposite (180 degrees) from positive
    neg_bar_angle = (bar_angle + 180) % 360
    neg_position_params = get_position_params_relative_to_bar(neg_bar_angle, negative_angle, pad)

    # Construct Altair expressions from the positive and negative position parameters
    datum = f"datum['{score_col}']"
    params = {}

    for param in ["align", "baseline"]:
        if pos_position_params[param] == neg_position_params[param]:
            params[param] = pos_position_params[param]
        else:
            params[param] = alt.expr(
                f"{datum} >= 0 ? '{pos_position_params[param]}' : '{neg_position_params[param]}'"
            )

    for param in ["dx", "dy"]:
        if abs(pos_position_params[param]) > 0.01 or abs(neg_position_params[param]) > 0.01:
            pos_rounded_param = round(pos_position_params[param], 3)
            neg_rounded_param = round(neg_position_params[param], 3)
            params[param] = alt.expr(f"{datum} >= 0 ? {pos_rounded_param} : {neg_rounded_param}")

    return params


def get_position_params_relative_to_bar(
    bar_angle: float, rotation: float, pad: float
) -> dict[str, float]:
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
    relative_angle_rad = math.radians((bar_angle - rotation) % 360)
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
