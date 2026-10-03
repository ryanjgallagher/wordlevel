"""Generate the cover word shift graphs shown on the home page and in the README.

The comparison itself is defined in `docs/snippets/index.py`, which the home page also includes as
its example code. This script runs that snippet and saves the resulting chart, so the code a reader
sees and the image beneath it cannot drift apart.

Unlike the rest of the documentation's charts, these are committed static files rather than
build-time renders: the README cannot execute snippets, and the home page should not depend on a
lexicon download at build time.

Two variants are written, one per color scheme, which the home page selects between with the
`#only-light` / `#only-dark` suffix. The light variant is the chart exactly as the snippet builds
it. The dark variant restates the same call with the theme's colors; see `_dark_chart` for why the
bar palette cannot simply be reused.

Usage:
    uv run python -m docs.scripts.generate_cover_word_shift_graph
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import altair as alt

from wordlevel.configs import ShiftBarConfig
from wordlevel.constants import COUNTERACTED_SUFFIX, SHIFT_BASE_SCORE_TYPES


REPO_ROOT = Path(__file__).resolve().parents[2]
SNIPPET = REPO_ROOT / "docs" / "snippets" / "index.py"
IMAGE_DIR = REPO_ROOT / "docs" / "assets" / "images"
LIGHT_OUTPUT = IMAGE_DIR / "cover_word_shift_graph.svg"
DARK_OUTPUT = IMAGE_DIR / "cover_word_shift_graph_dark.svg"

# The theme's "slate" scheme, resolved from its `hsla(225deg, …)` definitions and composited over
# the page background where the theme states an alpha. The text matches the white the header uses
# rather than the dimmer body foreground, so the labels stay legible over the rank grid lines.
DARK_BACKGROUND = "#0b0c0f"  # --md-default-bg-color
DARK_TEXT = "#ffffff"
DARK_MUTED_TEXT = "#b6b6b7"  # the header's 70% white, composited over the background
DARK_LINE = "#505155"  # --md-default-fg-color--lighter

# Each default `*_counteracted` color is its base color blended this far toward white: the faded
# look is a fade into the page, not a color in its own right. Against a dark page the same fade has
# to run toward the dark background instead, or the faded bars come out brighter than the bars they
# are meant to recede behind.
COUNTERACTED_FADE = 0.65


def _blend(color: str, toward: str, fraction: float) -> str:
    start = [int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
    end = [int(toward.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(
        f"{round(start[i] + fraction * (end[i] - start[i])):02x}" for i in range(3)
    )


def _dark_bar_config() -> dict[str, str]:
    """Re-fade every `*_counteracted` bar color against the dark background."""
    defaults = ShiftBarConfig()
    return {
        f"{name}{COUNTERACTED_SUFFIX}": _blend(
            getattr(defaults, name), DARK_BACKGROUND, COUNTERACTED_FADE
        )
        for name in SHIFT_BASE_SCORE_TYPES
    }


def _dark_chart(catalog: Any, comparison: str) -> alt.LayerChart | alt.VConcatChart:
    chart = catalog.plot.shift(
        comparison,
        show_totals=True,
        bar_config=_dark_bar_config(),
        label_config={"color": DARK_TEXT},
        score_axis={
            "title_color": DARK_MUTED_TEXT,
            "label_color": DARK_MUTED_TEXT,
            "tick_color": DARK_LINE,
            "axis_color": DARK_LINE,
        },
        rank_axis={
            "title_color": DARK_MUTED_TEXT,
            "label_color": DARK_MUTED_TEXT,
            "tick_color": DARK_LINE,
            "axis_color": DARK_LINE,
            "grid_color": DARK_LINE,
        },
        border_config={"color": DARK_LINE},
        zero_line_config={"color": DARK_LINE},
        total_shift_config={"total": DARK_MUTED_TEXT},
    )
    # The chart background is Altair's, not one of wordlevel's plotting options.
    return chart.properties(background=DARK_BACKGROUND)


def main() -> None:
    namespace = runpy.run_path(str(SNIPPET), run_name="__main__")
    catalog = namespace["cl"]
    (comparison,) = catalog.comparisons

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    namespace["chart"].save(LIGHT_OUTPUT)
    _dark_chart(catalog, comparison).save(DARK_OUTPUT)

    for output in (LIGHT_OUTPUT, DARK_OUTPUT):
        print(f"Wrote {output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
