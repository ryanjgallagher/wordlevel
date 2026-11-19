from typing import Any

import altair as alt


def to_altair_optional(val: Any) -> Any:
    """
    Converts `None` to `alt.Undefined`, so that Altair falls back to its own default for an option
    that is not set. Any other value is returned unchanged.
    """
    if val is None:
        return alt.Undefined
    else:
        return val
