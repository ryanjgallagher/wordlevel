from typing import cast

import altair as alt
import polars as pl
import pytest

from wordlevel.configs import BarConfig, LabelConfig, RankAxisConfig, ShiftColorConfig
from wordlevel.constants import DEFAULT_SCORE_AXIS_PIXEL_SIZE
from wordlevel.plots.core.shift.bars import (
    InternalBarConfig,
    InternalScoreAxisConfig,
)
from wordlevel.plots.core.shift.border import InternalBorderConfig
from wordlevel.plots.core.shift.shift import shift


def test_shift_returns_layer_chart(
    make_mock_df, make_shift_configs, score_col, rank_col, token_col, score_type_col
):
    """ """
    chart = shift(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        score_type_col=score_type_col,
        width=100,
        **make_shift_configs(),
    )

    assert isinstance(chart, alt.LayerChart)
    # shift_borders groups the base components (bars, labels, scales, zero line) into chart.layer[0]
    # and appends the border-edge layers after it.
    base = chart.layer[0]
    assert isinstance(base, alt.LayerChart)
    assert len(base.layer) == 4
    # The scales layer (a LayerChart of 2 rules) sits at index 2 of the base group
    assert isinstance(base.layer[2], alt.LayerChart)


@pytest.mark.parametrize("vertical", [True, False])
def test_shift_default_dimensions(
    make_mock_df, make_shift_configs, score_col, rank_col, token_col, score_type_col, vertical
):
    """
    The score-axis dimension defaults to DEFAULT_SCORE_AXIS_PIXEL_SIZE; the rank-axis dimension is
    the per-rank bar_width Step, so it scales per-rank rather than by total chart size.
    """
    chart = shift(
        make_mock_df(n=50),
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        score_type_col=score_type_col,
        max_rank=50,
        vertical=vertical,
        **make_shift_configs(),
    )

    spec = chart.to_dict()
    rank_step = {"step": RankAxisConfig().bar_width}

    if vertical:
        assert spec["width"] == DEFAULT_SCORE_AXIS_PIXEL_SIZE
        assert spec["height"] == rank_step
    else:
        assert spec["height"] == DEFAULT_SCORE_AXIS_PIXEL_SIZE
        assert spec["width"] == rank_step


@pytest.mark.parametrize("vertical", [True, False])
@pytest.mark.parametrize("bar_width", [8, 20])
def test_shift_bar_width_controls_rank_dimension(
    make_mock_df,
    make_shift_configs,
    score_col,
    rank_col,
    token_col,
    score_type_col,
    vertical,
    bar_width,
):
    """
    RankAxisConfig.bar_width sets the per-rank Step on the rank-axis dimension (height when
    vertical, width when horizontal) and flows through `shift`.
    """
    chart = shift(
        make_mock_df(n=50),
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        score_type_col=score_type_col,
        max_rank=50,
        vertical=vertical,
        **make_shift_configs(rank_axis_config=RankAxisConfig(bar_width=bar_width)),
    )

    spec = chart.to_dict()
    rank_dim = "height" if vertical else "width"
    assert spec[rank_dim] == {"step": bar_width}


@pytest.mark.parametrize("vertical", [True, False])
def test_shift_ignores_rank_axis_pixel_size(
    make_mock_df, make_shift_configs, score_col, rank_col, token_col, score_type_col, vertical
):
    """
    An explicit pixel size for the rank-axis dimension is ignored: `height` when vertical, `width`
    when horizontal (that dimension stays the bar_width Step). The score-axis dimension is honored.
    """
    chart = shift(
        make_mock_df(n=50),
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        score_type_col=score_type_col,
        max_rank=50,
        vertical=vertical,
        height=999,
        width=777,
        **make_shift_configs(),
    )

    spec = chart.to_dict()
    rank_step = {"step": RankAxisConfig().bar_width}

    if vertical:
        assert spec["height"] == rank_step  # explicit height ignored
        assert spec["width"] == 777  # explicit width honored (score axis)
    else:
        assert spec["width"] == rank_step  # explicit width ignored
        assert spec["height"] == 999  # explicit height honored (score axis)


@pytest.mark.parametrize("max_rank", [11, 40])
def test_shift_max_rank_filter(
    make_mock_df, make_shift_configs, score_col, rank_col, token_col, score_type_col, max_rank
):
    """
    Check that the wrapper filters data by max_rank before creating layers
    """
    chart = shift(
        make_mock_df(n=50),
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        score_type_col=score_type_col,
        max_rank=max_rank,
        width=100,
        **make_shift_configs(),
    )

    base = chart.layer[0]
    bars_data = cast(pl.DataFrame, base.layer[0].data)
    assert bars_data[rank_col].max() == max_rank

    labels_data = cast(pl.DataFrame, base.layer[1].data)
    assert labels_data[rank_col].max() == max_rank
    assert len(labels_data) == max_rank


def test_config_propagation(
    make_mock_df, make_shift_configs, score_col, rank_col, token_col, score_type_col
):
    """
    Check configs passed to the wrapper affect the final chart
    """
    custom_rank_title = "Custom Rank Title"
    custom_score_title = "Custom Score Title"
    custom_label_color = "cyan"

    chart = shift(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        score_type_col=score_type_col,
        width=100,
        vertical=True,
        **make_shift_configs(
            rank_axis_config=RankAxisConfig(title=custom_rank_title, grid=False),
            score_axis_config=InternalScoreAxisConfig(title=custom_score_title, grid=True),
            bar_config=InternalBarConfig.from_shift_colors(
                config=ShiftColorConfig(more_of_positive="magenta"), bar_config=BarConfig()
            ),
            label_config=LabelConfig(color=custom_label_color),
        ),
    )

    # The base components (bars, labels, ...) are grouped in layer[0]; borders follow
    base_layers = chart.to_dict()["layer"][0]["layer"]

    # Check bars layer
    bars_encoding = base_layers[0]["encoding"]

    assert bars_encoding["y"]["axis"]["title"] == custom_rank_title
    assert bars_encoding["y"]["axis"]["grid"] is False

    assert bars_encoding["x"]["axis"]["title"] == custom_score_title
    assert bars_encoding["x"]["axis"]["grid"] is True

    assert "magenta" in bars_encoding["color"]["scale"]["range"]

    # Check labels layer
    label_layer = base_layers[1]

    assert label_layer["mark"]["color"] == custom_label_color


def test_shift_border_config_propagation(
    make_mock_df, make_shift_configs, score_col, rank_col, token_col, score_type_col
):
    """
    The border config reaches both the bars (native score-axis domain) and the border framing
    (opposite edges)
    """
    chart = shift(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        score_type_col=score_type_col,
        width=100,
        vertical=True,
        **make_shift_configs(
            border_config=InternalBorderConfig(
                width=1,
                draw_score_axis=False,
                draw_border_opposite_rank_axis=True,
            )
        ),
    )

    bars_encoding = chart.to_dict()["layer"][0]["layer"][0]["encoding"]
    assert bars_encoding["x"]["axis"]["domain"] is False
    # chart.layer[0] is the base group; chart.layer[1:] are the border edges
    assert len(chart.layer[1:]) == 1


def test_shift_score_axis_opposite(
    make_mock_df, make_shift_configs, score_col, rank_col, token_col, score_type_col
):
    """`score_axis_opposite` moves a horizontal chart's score axis to the right"""
    chart = shift(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        score_type_col=score_type_col,
        height=100,
        vertical=False,
        score_axis_opposite=True,
        **make_shift_configs(),
    )

    bars_encoding = chart.to_dict()["layer"][0]["layer"][0]["encoding"]
    assert bars_encoding["y"]["axis"]["orient"] == "right"
