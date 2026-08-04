from typing import cast

import altair as alt
import polars as pl
import pytest

from wordlevel.configs import LabelConfig, RankAxisConfig, ShiftBarConfig
from wordlevel.constants import DEFAULT_SCORE_AXIS_PIXEL_SIZE
from wordlevel.core.catalog.plot.shift.render.bars import (
    InternalBarConfig,
    InternalScoreAxisConfig,
)
from wordlevel.core.catalog.plot.shift.render.shift import shift


def test_shift_returns_layer_chart(make_mock_df, score_col, rank_col, token_col, score_type_col):
    """ """
    chart = shift(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        score_type_col=score_type_col,
        width=100,
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
    make_mock_df, score_col, rank_col, token_col, score_type_col, vertical
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
    make_mock_df, score_col, rank_col, token_col, score_type_col, vertical, bar_width
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
        rank_axis=RankAxisConfig(bar_width=bar_width),
    )

    spec = chart.to_dict()
    rank_dim = "height" if vertical else "width"
    assert spec[rank_dim] == {"step": bar_width}


@pytest.mark.parametrize("vertical", [True, False])
def test_shift_ignores_rank_axis_pixel_size(
    make_mock_df, score_col, rank_col, token_col, score_type_col, vertical
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
    make_mock_df, score_col, rank_col, token_col, score_type_col, max_rank
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
    )

    base = chart.layer[0]
    bars_data = cast(pl.DataFrame, base.layer[0].data)
    assert bars_data[rank_col].max() == max_rank

    labels_data = cast(pl.DataFrame, base.layer[1].data)
    assert labels_data[rank_col].max() == max_rank
    assert len(labels_data) == max_rank


def test_config_propagation(make_mock_df, score_col, rank_col, token_col, score_type_col):
    """
    Check configs passed to the wrapper affect the final chart
    """
    custom_rank_title = "Custom Rank Title"
    custom_score_title = "Custom Score Title"
    custom_bar_colors = {"more_of_positive_counteracted": "magenta"}
    custom_label_color = "cyan"

    for config_style in ["config", "dict"]:
        if config_style == "config":
            rank_config = RankAxisConfig(title=custom_rank_title, grid=False)
            score_config = InternalScoreAxisConfig(title=custom_score_title, grid=True)
            shift_bar_config = ShiftBarConfig(**custom_bar_colors)
            label_config = LabelConfig(color=custom_label_color)
        else:
            rank_config = {"title": custom_rank_title, "grid": False}
            score_config = {"title": custom_score_title, "grid": True}
            shift_bar_config = custom_bar_colors
            label_config = {"color": custom_label_color}

        chart = shift(
            make_mock_df(),
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            score_type_col=score_type_col,
            width=100,
            vertical=True,
            rank_axis=rank_config,
            score_axis=score_config,
            bar_config=InternalBarConfig.from_shift_bar_config(shift_bar_config),
            label_config=label_config,
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
