import altair as alt
import polars as pl
import pytest

from wordlevel.configs import BorderConfig, RankAxisConfig, ScoreAxisConfig
from wordlevel.core.catalog.plot.shift.render.bars import shift_bars
from wordlevel.core.catalog.plot.shift.render.border import InternalBorderConfig, shift_borders


@pytest.fixture
def df_mock():
    return pl.DataFrame(
        {
            "rank": [1, 2, 3],
            "score": [1.0, -1.0, 0.5],
            "score_type": ["reference", "reference", "comparison"],
        }
    )


@pytest.fixture
def base_chart(df_mock):
    return shift_bars(
        df_mock,
        score_col="score",
        rank_col="rank",
        score_type_col="score_type",
    )


def _border_axis(border_chart):
    """The single (x or y) axis spec carried by a border-edge chart."""
    encoding = border_chart.to_dict()["encoding"]
    return (encoding.get("x") or encoding.get("y"))["axis"]


@pytest.mark.parametrize(
    "vertical, draw_opposite_rank, draw_opposite_score, expected_orients",
    [
        # Vertical: rank's opposite edge is on the right, score's is on top
        (True, False, False, set()),
        (True, True, False, {"right"}),
        (True, False, True, {"top"}),
        (True, True, True, {"right", "top"}),
        # Horizontal: the orientations swap — rank's opposite is on top, score's is on the right
        (False, False, False, set()),
        (False, True, False, {"top"}),
        (False, False, True, {"right"}),
        (False, True, True, {"top", "right"}),
    ],
)
def test_opposite_edges_drawn(
    base_chart, vertical, draw_opposite_rank, draw_opposite_score, expected_orients
):
    """`shift_borders` adds one border-edge layer per requested opposite edge, and no others."""
    chart = shift_borders(
        [base_chart],
        rank_col="rank",
        score_col="score",
        rank_axis=RankAxisConfig(),
        score_axis=ScoreAxisConfig(),
        border=InternalBorderConfig(
            width=1,
            draw_border_opposite_rank_axis=draw_opposite_rank,
            draw_border_opposite_score_axis=draw_opposite_score,
        ),
        vertical=vertical,
    )

    # chart.layer[0] is the base group; chart.layer[1:] are the border edges.
    border_orients = {_border_axis(edge)["orient"] for edge in chart.layer[1:]}
    assert border_orients == expected_orients


def test_border_edges_color_and_width(base_chart):
    """Both opposite edges take their domain color and width from the border config."""
    chart = shift_borders(
        [base_chart],
        rank_col="rank",
        score_col="score",
        rank_axis=RankAxisConfig(),
        score_axis=ScoreAxisConfig(),
        border=InternalBorderConfig(
            color="purple",
            width=3,
            draw_border_opposite_rank_axis=True,
            draw_border_opposite_score_axis=True,
        ),
        vertical=True,
    )

    edges = chart.layer[1:]
    assert len(edges) == 2
    for edge in edges:
        axis = _border_axis(edge)
        assert axis["domainColor"] == "purple"
        assert axis["domainWidth"] == 3


def test_border_edges_use_axis_color_over_border_color(base_chart):
    """Each edge's domain color comes from its axis's `axis_color`, overriding the border color."""
    chart = shift_borders(
        [base_chart],
        rank_col="rank",
        score_col="score",
        rank_axis=RankAxisConfig(axis_color="red"),
        score_axis=ScoreAxisConfig(axis_color="green"),
        border=InternalBorderConfig(
            color="black",
            width=1,
            draw_border_opposite_rank_axis=True,
            draw_border_opposite_score_axis=True,
        ),
        vertical=True,
    )

    # Vertical: rank's opposite edge is on the right, score's on top. Neither should be "black".
    colors_by_orient = {
        _border_axis(edge)["orient"]: _border_axis(edge)["domainColor"] for edge in chart.layer[1:]
    }
    assert colors_by_orient == {"right": "red", "top": "green"}


def test_border_edges_are_non_interactive(base_chart):
    """The border-edge charts are invisible (opacity 0) and carry no tooltip / hover target."""
    chart = shift_borders(
        [base_chart],
        rank_col="rank",
        score_col="score",
        rank_axis=RankAxisConfig(),
        score_axis=ScoreAxisConfig(),
        border=InternalBorderConfig(
            width=1,
            draw_border_opposite_rank_axis=True,
            draw_border_opposite_score_axis=True,
        ),
        vertical=True,
    )

    for edge in chart.layer[1:]:
        spec = edge.to_dict()
        assert spec["mark"]["opacity"] == 0
        assert spec["encoding"]["tooltip"]["value"] is None


def test_base_layers_grouped_not_flattened(base_chart, df_mock):
    """
    The input charts are nested together under `chart.layer[0]`, not flattened among the border
    edges
    """
    extra = alt.Chart(df_mock).mark_text().encode(x="score:Q", y="rank:O")

    chart = shift_borders(
        [base_chart, extra],
        rank_col="rank",
        score_col="score",
        rank_axis=RankAxisConfig(),
        score_axis=ScoreAxisConfig(),
        border=InternalBorderConfig(
            width=1,
            draw_border_opposite_rank_axis=True,
            draw_border_opposite_score_axis=True,
        ),
        vertical=True,
    )

    base_group = chart.layer[0]
    assert isinstance(base_group, alt.LayerChart)
    assert len(base_group.layer) == 2  # both inputs grouped together, not flattened
    assert len(chart.layer) - 1 == 2  # two border edges layered after the group


@pytest.mark.parametrize("width, expected_width", [(None, 4), (2, 2)])
def test_from_config_width_override(width, expected_width):
    """
    `from_config` takes its width from the user-facing config, unless a caller overrides it to draw
    an edge that a zero-width border box would otherwise leave invisible
    """
    border = InternalBorderConfig.from_config(BorderConfig(width=4), width=width)

    assert border.width == expected_width
