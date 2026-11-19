import altair as alt

from wordlevel.configs import BorderConfig, ZeroLineConfig
from wordlevel.plots.core.shift.border import _TRANSLATE_PX
from wordlevel.plots.core.utils import to_altair_optional


def zero_line(
    zero_line_config: ZeroLineConfig,
    border_config: BorderConfig,
    vertical: bool = True,
) -> alt.Chart:
    """
    Plots a guide line at `0` on the score axis.

    The line's color comes from `zero_line_config`, falling back to `border_config` so it matches
    the chart frame when not set. A width of `0` draws no visible line.
    """
    color = zero_line_config.color if zero_line_config.color is not None else border_config.color

    # The score axis (and its zero tick) is nudged a half pixel for crisp registration with the
    # frame (see `border._TRANSLATE_PX`). The zero line is a mark, not an axis, so it carries the
    # same offset explicitly — otherwise it would sit a half pixel off the zero tick.
    if vertical:
        rule = alt.Chart().mark_rule(
            color=to_altair_optional(color),
            strokeWidth=zero_line_config.width,
            xOffset=_TRANSLATE_PX,
        )
        return rule.encode(x=alt.datum(0))
    else:
        rule = alt.Chart().mark_rule(
            color=to_altair_optional(color),
            strokeWidth=zero_line_config.width,
            yOffset=_TRANSLATE_PX,
        )
        return rule.encode(y=alt.datum(0))
