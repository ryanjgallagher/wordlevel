import altair as alt

from wordlevel.configs import BorderConfig, ZeroLineConfig
from wordlevel.plots.core.shift.border import _TRANSLATE_PX
from wordlevel.plots.core.utils import to_altair_optional


def zero_line(
    zero_line_config: ZeroLineConfig | dict | None = None,
    border_config: BorderConfig | dict | None = None,
    vertical: bool = True,
) -> alt.Chart:
    """
    Plots a guide line at `0` on the score axis.

    The line's color comes from `zero_line_config`, falling back to `border_config` so it matches
    the chart frame when not set. A width of `0` (the default) draws no visible line.
    """
    if isinstance(zero_line_config, dict):
        zero_line_config = ZeroLineConfig(**zero_line_config)
    elif zero_line_config is None:
        zero_line_config = ZeroLineConfig()

    if isinstance(border_config, dict):
        border_config = BorderConfig(**border_config)
    elif border_config is None:
        border_config = BorderConfig()

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
