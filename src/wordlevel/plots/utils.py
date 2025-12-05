from typing import Any

import altair as alt


def to_altair_optional(val: None | Any) -> None | Any:
    """ """
    if val is None:
        return alt.Undefined
    else:
        return val
