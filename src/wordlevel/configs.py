"""Configuration classes for wordlevel"""

from dataclasses import dataclass


DEFAULT_LABEL_SIZE: float = 11

DEFAULT_SCORE_AXIS_TITLE = "Contribution"
PERCENT_SCORE_AXIS_TITLE = "Percent contribution"

LABEL_SUFFIX = "_label"

DEFAULT_BAR_OUTLINE_COLOR = "#888"
DEFAULT_BAR_OUTLINE_WIDTH: float = 0.1

DEFAULT_GRAND_TOTAL_COLOR = "#888"
DEFAULT_ATTRIBUTION_COLORS = {
    "reference": "#F09E2E",
    "comparison": "#016FBA",
    "reinforce": "#016FBA",
    "offset": "#E6E6E6",
}

NON_SCORE_TYPE_FIELDS = frozenset({"grand_total", "counteracted_opacity"})


@dataclass(kw_only=True)
class BarConfig:
    """
    Configuration for the bars of a bar or word shift plot.

    For the color of each bar, see
    [`BarColorConfig`][wordlevel.configs.BarColorConfig] and
    [`ShiftColorConfig`][wordlevel.configs.ShiftColorConfig].

    Attributes:
        outline_color:
            Color of the outline around each bar
        outline_width:
            Width of the outline around each bar, in pixels. `0` draws no outline
    """

    outline_color: str = DEFAULT_BAR_OUTLINE_COLOR
    outline_width: float = DEFAULT_BAR_OUTLINE_WIDTH


@dataclass(kw_only=True)
class BarColorConfig:
    """
    Configuration for the bar colors of a bar plot.

    Contributions are split into two groups, and which pair of colors applies depends on how the
    comparison attributes them. Measures attributed by frequency or rank use `reference` and
    `comparison`, for whether a token is more common in the reference or the comparison corpus.
    Measures attributed by score sign use `reinforce` and `offset`, for whether a token's score
    makes the overall difference larger or smaller. Setting a color for the pair that does not
    apply to the plotted comparison raises an error naming the pair that does.

    Attributes:
        reference:
            Color of bars attributed to the reference corpus. If `None`, `"#F09E2E"` is used
        comparison:
            Color of bars attributed to the comparison corpus. If `None`, `"#016FBA"` is used
        reinforce:
            Color of bars where the token reinforces the overall difference, making its magnitude
            larger. If `None`, `"#016FBA"` is used
        offset:
            Color of bars where the token offsets the overall difference, making its magnitude
            smaller. If `None`, `"#E6E6E6"` is used
        grand_total:
            Color of the grand total bar in the totals panels
    """

    reference: str | None = None
    comparison: str | None = None
    reinforce: str | None = None
    offset: str | None = None
    grand_total: str = DEFAULT_GRAND_TOTAL_COLOR


@dataclass(kw_only=True)
class ShiftColorConfig:
    """
    Configuration for the bar colors of a word shift plot.

    The bars indicate the kind of contribution each token makes. The prefixes `more_of_*` and
    `less_of_*` indicate whether a token was used more or less in the comparison corpus versus the
    reference corpus. The suffixes `*_positive` and `*_negative` indicate whether a token is
    relatively positive or negative compared to the specified reference score. The bars
    `higher_score` and `lower_score` indicate whether the token's score is higher or lower in the
    comparison corpus. Each of these six bars also has a `*_counteracted` variant, for when the two
    base bars are opposite signs and partially cancel out. A counteracted bar is drawn in its base
    bar's color, faded by `counteracted_opacity`, so it is not colored separately.

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
        counteracted_opacity:
            Opacity of the `*_counteracted` bars, as a fraction between 0 and 1. Each one is drawn
            in its base bar's color at this opacity
        grand_total:
            Color of the grand total bar in the totals panels
    """

    more_of_negative: str = "#2F7CCE"
    more_of_positive: str = "#FFFF80"
    less_of_positive: str = "#FDFFD2"
    less_of_negative: str = "#C4CAFC"
    lower_score: str = "#9E75B7"
    higher_score: str = "#FECC5D"
    counteracted_opacity: float = 0.345
    grand_total: str = DEFAULT_GRAND_TOTAL_COLOR


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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
        borrow_symbol:
            A symbol appended to the end of a token's label indicating that it "borrowed" a score
            across corpora. If `None`, no symbol is appended. For detail on borrowing, see
            `scored_in_one_lexicon` in
            [lexicon scoring][wordlevel.comparison.score.Scorer.lexicon]
    """

    pad: float = 5
    color: str = "black"
    size: float | None = DEFAULT_LABEL_SIZE
    weight: str | int = "normal"
    rotation: float | None = None
    mirror_rotation: bool = True
    on_bar_ends: bool = True
    font_width_to_height_ratio: float = 0.8
    borrow_symbol: str | None = "*"


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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
    width: float = 1.0


@dataclass(kw_only=True)
class TotalsConfig:
    """
    Configuration for the totals panels of a bar or word shift plot.

    The totals panels show one cumulative-contribution bar per contribution type, plus a grand
    total bar summing them all. Each bar is labeled with the text of its contribution type, so only
    the labels matching the plotted comparison apply: `reference_label` and `comparison_label` for
    contributions attributed by frequency or rank, `reinforce_label` and `offset_label` for those
    attributed by score sign, and the six remaining labels for a word shift plot. The labels that do
    not apply are ignored. Every bar's color, the grand total's included, comes from the color
    config.

    Attributes:
        grand_total_label:
            Label for the grand total bar
        reference_label:
            Label for the contribution attributed to the reference corpus
        comparison_label:
            Label for the contribution attributed to the comparison corpus
        reinforce_label:
            Label for the contribution that reinforces the overall difference
        offset_label:
            Label for the contribution that offsets the overall difference
        more_of_negative_label:
            Label for the `more_of_negative` contribution
        more_of_positive_label:
            Label for the `more_of_positive` contribution
        less_of_positive_label:
            Label for the `less_of_positive` contribution
        less_of_negative_label:
            Label for the `less_of_negative` contribution
        lower_score_label:
            Label for the `lower_score` contribution
        higher_score_label:
            Label for the `higher_score` contribution
        grand_total_outline_color:
            Color of the outline around the grand total bar. If `None`, `BarConfig.outline_color`
            is used
        grand_total_outline_width:
            Width of the outline around the grand total bar, in pixels. If `None`,
            `BarConfig.outline_width` is used
    """

    grand_total_label: str = "Total"
    reference_label: str = "Reference"
    comparison_label: str = "Comparison"
    reinforce_label: str = "Reinforces"
    offset_label: str = "Offsets"
    more_of_negative_label: str = "−↑"
    more_of_positive_label: str = "+↑"
    less_of_positive_label: str = "+↓"
    less_of_negative_label: str = "−↓"
    lower_score_label: str = "▽"
    higher_score_label: str = "△"
    grand_total_outline_color: str | None = None
    grand_total_outline_width: float | None = None
