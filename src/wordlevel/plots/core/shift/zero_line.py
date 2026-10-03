import altair as alt

from wordlevel.configs import BorderConfig, ZeroLineConfig
from wordlevel.plots.core.shift.border import _TRANSLATE_PX
from wordlevel.plots.core.utils import to_altair_optional


def zero_line(
    config: ZeroLineConfig | dict | None = None,
    border: BorderConfig | dict | None = None,
    vertical: bool = True,
) -> alt.Chart:
    """
    Plots a guide line at `0` on the score axis.

    The line's color comes from `config`, falling back to `border` so it matches the chart frame
    when not set. A width of `0` (the default) draws no visible line.
    """
    if isinstance(config, dict):
        config = ZeroLineConfig(**config)
    elif config is None:
        config = ZeroLineConfig()

    if isinstance(border, dict):
        border = BorderConfig(**border)
    elif border is None:
        border = BorderConfig()

    color = config.color if config.color is not None else border.color

    # The score axis (and its zero tick) is nudged a half pixel for crisp registration with the
    # frame (see `border._TRANSLATE_PX`). The zero line is a mark, not an axis, so it carries the
    # same offset explicitly — otherwise it would sit a half pixel off the zero tick.
    if vertical:
        rule = alt.Chart().mark_rule(
            color=to_altair_optional(color), strokeWidth=config.width, xOffset=_TRANSLATE_PX
        )
        return rule.encode(x=alt.datum(0))
    else:
        rule = alt.Chart().mark_rule(
            color=to_altair_optional(color), strokeWidth=config.width, yOffset=_TRANSLATE_PX
        )
        return rule.encode(y=alt.datum(0))
