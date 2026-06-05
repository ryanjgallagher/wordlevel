"""Configuration classes for wordlevel"""

from dataclasses import dataclass, fields
from types import SimpleNamespace


DEFAULT_LABEL_SIZE: float = 11


@dataclass
class BarConfig:
    """
    Configuration for the bar colors of a bar plot.

    The bars are split by which corpus a token's contribution is attributed to: the reference or the
    comparison

    Attributes:
        reference:
            Color of bars attributed to the reference corpus
        comparison:
            Color of bars attributed to the comparison corpus
    """

    reference: str = "#FECC5D"
    comparison: str = "#2F7CCE"


@dataclass
class AlignmentBarConfig:
    """
    Configuration for the bar colors of a bar plot scored by a score-sign measure.

    For measures like
    [lexicon-weighted averages][wordlevel.api.comparison.score.Scorer.lexicon],
    [Shannon entropy][wordlevel.api.comparison.score.Scorer.shannon_entropy],
    [Tsallis entropy][wordlevel.api.comparison.score.Scorer.tsallis_entropy], and the
    [Kullback-Leibler divergence][wordlevel.api.comparison.score.Scorer.kullback_leibler_divergence],
    a token cannot be cleanly attributed to one corpus. Instead, the bars are split by whether a
    token's contribution reinforces the overall difference between the corpora (pushing it further
    in its realized direction) or offsets it (pulling it back).

    Attributes:
        reinforce:
            Color of bars where the token reinforces the overall difference, making its magnitude
            larger
        offset:
            Color of bars where the token offsets the overall difference, making its magnitude
            smaller
    """

    reinforce: str = "#2F7CCE"
    offset: str = "#FECC5D"


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


SHIFT_BAR_NAMES = SimpleNamespace(**{f.name: f.name for f in fields(ShiftBarConfig)})
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
            Label for the score axis. If `None`, the title is hidden
        title_fontsize:
            Font size of the title, in pixels. If `None`, Altair's default is used
        label_fontsize:
            Font size of the axis tick labels, in pixels. If `None`, Altair's default is used
        grid:
            Whether to draw gridlines along the score axis
    """

    title: str | None = "Contribution"
    title_fontsize: float | None = None
    label_fontsize: float | None = None
    grid: bool = False


@dataclass
class RankAxisConfig:
    """
    Configuration for the rank axis of a bar or word shift plot.

    The rank axis is the axis along which tokens are listed in order of contribution.

    Attributes:
        title:
            Label for the rank axis. If `None`, the title is hidden
        title_fontsize:
            Font size of the title, in pixels. If `None`, Altair's default is used
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
        grid:
            Whether to draw gridlines along the rank axis
    """

    title: str | None = "Rank"
    title_fontsize: float | None = None
    label_fontsize: float | None = None
    label_tick_values: list[int] | None = None
    label_tick_step: int | None = 5
    label_first_tick: bool = True
    grid: bool = True


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
