"""Configuration classes for wordlevel"""

from dataclasses import dataclass, fields
from types import SimpleNamespace


DEFAULT_LABEL_SIZE: float = 11

DEFAULT_SCORE_AXIS_TITLE = "Contribution"
PERCENT_SCORE_AXIS_TITLE = "Percent contribution"

SYMBOL_SUFFIX = "_symbol"


@dataclass
class BarConfig:
    """
    Configuration for the bar colors of token contributions attributed by frequency or rank.

    The contributions (and bars) are attributed by whether a token is more common in the reference
    or the comparison.

    Attributes:
        reference:
            Color of bars attributed to the reference corpus
        comparison:
            Color of bars attributed to the comparison corpus
        reference_symbol:
            Symbol labeling the reference contribution in the totals panels
        comparison_symbol:
            Symbol labeling the comparison contribution in the totals panels
    """

    reference: str = "#FECC5D"
    comparison: str = "#2F7CCE"
    reference_symbol: str = "Reference"
    comparison_symbol: str = "Comparison"


@dataclass
class ScoreSignBarConfig:
    """
    Configuration for the bar colors of token contributions attributed by score sign.

    The contributions (and bars) are attributed by whether a token's score is positive or negative,
    and whether that score reinforces or offsets the overall corpus-level difference between the
    reference and comparison.

    Attributes:
        reinforce:
            Color of bars where the token reinforces the overall difference, making its magnitude
            larger
        offset:
            Color of bars where the token offsets the overall difference, making its magnitude
            smaller
        reinforce_symbol:
            Symbol labeling the reinforcing contribution in the totals panels
        offset_symbol:
            Symbol labeling the offsetting contribution in the totals panels
    """

    reinforce: str = "#2F7CCE"
    offset: str = "#FECC5D"
    reinforce_symbol: str = "Reinforces"
    offset_symbol: str = "Offsets"


@dataclass
class ShiftBarConfig:
    """
    Configuration for the bar colors of a word shift plot.

    The bars indicate the kind of contribution each token makes. The prefixes `more_of_*` and
    `less_of_*` indicate whether a token was used more or less in the comparison corpus versus the
    referenc corpus. The suffixes `*_positive` and `*_negative` indicate whether a token is
    relatively positive or negative compared to the specified reference score. The bars
    `higher_score` and `lower_score` indicate whether the token's score is higher or lower in the
    comparison corpus. Each of these six primary bars also have a `*_counteracted` variant for when
    the two base bars are opposite signs and partially cancel out

    Attributes:
        more_of_negative:
            Color of bars where the token is used more in the comparison and has a relatively
            negative score
        more_of_positive:
            Color of bars where the token is used more in the comparison and has a relatively
            positive score
        less_of_positive:
            Color of bars where the token is used less in the comparison and has a relatively
            positive score
        less_of_negative:
            Color of bars where the token is used less in the comparison and has a relatively
            negative score
        lower_score:
            Color of bars where the token's score is lower in the comparison than the reference
        higher_score:
            Color of bars where the token's score is higher in the comparison than the reference
        lower_score_counteracted:
            Color of bars where the `lower_score` contribution is partially counteracted by the
            other bar
        higher_score_counteracted:
            Color of bars where the `higher_score` contribution is partially counteracted by the
            other bar
        more_of_negative_counteracted:
            Color of bars where the `more_of_negative` contribution is partially counteracted by
            other bar
        more_of_positive_counteracted:
            Color of bars where the `more_of_positive` contribution is partially counteracted by
            other bar
        less_of_positive_counteracted:
            Color of bars where the `less_of_positive` contribution is partially counteracted by
            other bar
        less_of_negative_counteracted:
            Color of bars where the `less_of_negative` contribution is partially counteracted by
            other bar
        more_of_negative_symbol:
            Symbol labeling the `more_of_negative` contribution in the totals panels
        more_of_positive_symbol:
            Symbol labeling the `more_of_positive` contribution in the totals panels
        less_of_positive_symbol:
            Symbol labeling the `less_of_positive` contribution in the totals panels
        less_of_negative_symbol:
            Symbol labeling the `less_of_negative` contribution in the totals panels
        lower_score_symbol:
            Symbol labeling the `lower_score` contribution in the totals panels
        higher_score_symbol:
            Symbol labeling the `higher_score` contribution in the totals panels
    """

    more_of_negative: str = "#2F7CCE"
    more_of_positive: str = "#FFFF80"
    less_of_positive: str = "#FDFFD2"
    less_of_negative: str = "#C4CAFC"
    lower_score: str = "#9E75B7"
    higher_score: str = "#FECC5D"
    lower_score_counteracted: str = "#decee6"
    higher_score_counteracted: str = "#ffedc7"
    more_of_negative_counteracted: str = "#b6d1ee"
    more_of_positive_counteracted: str = "#ffffd4"
    less_of_positive_counteracted: str = "#fffff0"
    less_of_negative_counteracted: str = "#ebedff"
    more_of_negative_symbol: str = "−↑"
    more_of_positive_symbol: str = "+↑"
    less_of_positive_symbol: str = "+↓"
    less_of_negative_symbol: str = "−↓"
    lower_score_symbol: str = "▽"
    higher_score_symbol: str = "△"


SHIFT_BAR_NAMES = SimpleNamespace(
    **{f.name: f.name for f in fields(ShiftBarConfig) if not f.name.endswith(SYMBOL_SUFFIX)}
)
"""
Namespace of `ShiftBarConfig` field names as strings. Use this as the source of truth for shift type 
name strings in code that lives outside `ShiftBarConfig` (e.g., score plan aliases,
dataframe column names).
"""


@dataclass
class ScoreAxisConfig:
    """
    Configuration for the score axis of a bar or word shift plot.

    The score axis is the axis along which token contributions extend.

    Attributes:
        title:
            Label for the score axis. Defaults to `"Contribution"`, which becomes
            `"Percent contribution"` when the scores are shown as percentages (i.e. normalized).
            Set a different string to override, or `None` to hide the title
        title_color:
            Color of the title. If `None`, Altair's default is used
        title_fontsize:
            Font size of the title, in pixels. If `None`, Altair's default is used
        label_color:
            Color of the axis tick labels. If `None`, Altair's default is used
        label_fontsize:
            Font size of the axis tick labels, in pixels. If `None`, Altair's default is used
        tick_color:
            Color of the axis ticks. For the color of the tick labels, see `label_color`. If not
            `None`, overrides `BorderConfig.color`
        grid:
            Whether to draw grid lines along the score axis
        grid_color:
            Color of the grid lines
        axis_color:
            Color of the axis line. If not `None`, overrides `BorderConfig.color`
        width:
            Width of the axis, in pixels
    """

    title: str | None = DEFAULT_SCORE_AXIS_TITLE
    title_color: str | None = None
    title_fontsize: float | None = None
    label_color: str | None = None
    label_fontsize: float | None = None
    tick_color: str | None = None
    grid: bool = False
    grid_color: str | None = None
    axis_color: str | None = None
    width: float = 1


@dataclass
class RankAxisConfig:
    """
    Configuration for the rank axis of a bar or word shift plot.

    The rank axis is the axis along which tokens are listed in order of contribution.

    Attributes:
        bar_width:
            Amount of space, in pixels, that each rank occupies. This implicitly controls the
            height/width of the plot, depending on whether it is vertical/horizontal respectively
        bar_padding:
            Amount of space, given to padding between bars, as a fraction between 0 and 1. When `0`,
            the bars touch; when close to `1`, most of the visual space is padding (not bars). The
            bar thickness is `bar_width * (1 - bar_padding)`
        axis_padding:
            Amount of space given to adding space between the first and last bar and the axis.
            The space is a fraction between 0 and 1 of the `bar_width`. When `0`, the first and last
            bar sit flush against the rank axis; when `1`, padding equivalent to `bar_width` is
            added to the ends of the rank axis
        title:
            Label for the rank axis. If `None`, the title is hidden
        title_color:
            Color of the title. If `None`, Altair's default is used
        title_fontsize:
            Font size of the title, in pixels. If `None`, Altair's default is used
        label_color:
            Color of the axis tick labels. If `None`, Altair's default is used
        label_fontsize:
            Font size of the axis tick labels, in pixels. If `None`, Altair's default is used
        label_tick_values:
            Explicit list of rank values at which to place ticks. When set, takes precedence over
            `label_tick_step`. If `None`, ticks are placed according to `label_tick_step`
        label_tick_step:
            Spacing between rank tick marks (e.g. `5` places ticks at 5, 10, 15, …). If `None`,
            Altair's default tick placement is used. Ignored when `label_tick_values` is set
        label_first_tick:
            Whether to include a tick at rank 1 in addition to the ticks defined by
            `label_tick_step`. Has no effect when `label_tick_step` is `1` or when
            `label_tick_values` is set
        tick_color:
            Color of the axis ticks. For the color of the tick labels, see `label_color`. If not
            `None`, overrides `BorderConfig.color`
        grid:
            Whether to draw grid lines along the rank axis
        grid_color:
            Color of the grid lines
        axis_color:
            Color of the axis line. If not `None`, overrides `BorderConfig.color`
        width:
            Width of the axis, in pixels
    """

    bar_width: float = 14
    bar_padding: float = 0.1
    axis_padding: float = 0.5
    title: str | None = "Rank"
    title_color: str | None = None
    title_fontsize: float | None = None
    label_color: str | None = None
    label_fontsize: float | None = None
    label_tick_values: list[int] | None = None
    label_tick_step: int | None = 5
    label_first_tick: bool = True
    tick_color: str | None = None
    grid: bool = True
    grid_color: str | None = None
    axis_color: str | None = None
    width: float = 1


@dataclass
class LabelConfig:
    """
    Configuration for the token labels of a bar or word shift plot.

    Controls how token text is positioned, oriented, and styled relative to its bar.

    Attributes:
        pad:
            Padding between a label and the end of its bar, in pixels
        color:
            Font color of the label text
        size:
            Font size of the label text, in pixels. If `None`, Altair's default is used
        weight:
            Font weight of the label text. Accepts CSS font-weight strings (e.g. `"normal"`,
            `"bold"`) or numeric weights (e.g. `400`, `700`)
        rotation:
            Rotation of the label text, in degrees. If `None`, defaults are chosen based on the
            chart orientation (0° when the chart is vertical, 270° when horizontal)
        mirror_rotation:
            When `True`, labels on negative-direction bars are rotated by `-rotation` so they read
            in a complementary orientation to the positive labels. When `False`, all labels share
            the same `rotation`
        on_bar_ends:
            When `True`, labels sit at the tip of their bar. When `False`, labels sit at the
            score-axis baseline (the zero line)
        font_width_to_height_ratio:
            Assumed ratio of character width to character height for the label font. Used by
            autoscaling to estimate label widths from font size
    """

    pad: float = 5
    color: str = "black"
    size: float | None = DEFAULT_LABEL_SIZE
    weight: str | int = "normal"
    rotation: float | None = None
    mirror_rotation: bool = True
    on_bar_ends: bool = True
    font_width_to_height_ratio: float = 0.6


@dataclass
class BorderConfig:
    """
    Configuration for the border of the entire chart, regardless of axis.

    Attributes:
        color:
            Color of the border lines, tick marks, and zero line. `RankAxisConfig.axis_color`,
            `ScoreAxisConfig.axis_color`, and `ZeroLineConfig.color` take priority if specified
        width:
            Width of the border edges opposite the axes, in pixels. `width = 0` draws no border box
            (only the rank and score axes remain); `width > 0` closes the box at that width. The
            native rank and score axis widths are owned by `RankAxisConfig.width`  and
            `ScoreAxisConfig.width`
    """

    color: str | None = "#888"
    width: float = 0


@dataclass
class ZeroLineConfig:
    """
    Configuration for the zero line drawn at the score-axis origin.

    The zero line is a reference rule at a contribution of `0`, spanning the rank axis (running
    vertically through a vertical chart, horizontally through a horizontal one).

    Attributes:
        color:
            Color of the zero line. If `None`, falls back to `BorderConfig.color` so the line
            matches the chart frame
        width:
            Width of the zero line, in pixels. `0` draws no line
    """

    color: str | None = None
    width: float = 0


@dataclass
class TotalShiftConfig:
    """
    Configuration for the totals panels above a word shift plot.

    The totals panels show one cumulative-contribution bar per contribution type, plus a grand
    total bar summing them all. Per-type colors and symbols come from the bar config.

    Attributes:
        total:
            Color of the grand total bar
        total_symbol:
            Symbol labeling the grand total bar in the totals panels
    """

    total: str = "#888"
    total_symbol: str = "Σ"
