import pytest
import altair as alt

from wordlevel.plots.shift.shift import shift
from wordlevel.plots.shift.labels import LabelConfig
from wordlevel.plots.shift.bars import RankAxisConfig, ScoreAxisConfig, BarConfig


def test_shift_returns_layer_chart(make_mock_df, score_col, rank_col, label_col, score_type_col):
    """ """
    chart = shift(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        label_col=label_col,
        score_type_col=score_type_col,
        width=100,  # Required for autoscale default
    )

    assert isinstance(chart, alt.LayerChart)
    assert len(chart.layer) == 3


@pytest.mark.parametrize("autoscale, n_layers_expected", [(True, 3), (False, 2)])
def test_shift_autoscale_toggle(
    make_mock_df, score_col, rank_col, label_col, score_type_col, autoscale, n_layers_expected
):
    """
    Check that autoscale_axes=False removes the scaling layer
    """
    chart = shift(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        label_col=label_col,
        score_type_col=score_type_col,
        width=100,
        autoscale_axes=autoscale,
    )

    assert len(chart.layer) == n_layers_expected

    if autoscale:
        # The last layer should be the scales layer (which is a LayerChart of 2 rules)
        assert isinstance(chart.layer[2], alt.LayerChart)


def test_shift_dimension_validation(make_mock_df, score_col, rank_col, label_col, score_type_col):
    """
    Check that the wrapper propagates the ValueError from shift_scales when dimensions are
    missing
    """
    # Case 1: autoscale=True + Vertical + No Width -> Fail
    with pytest.raises(ValueError, match="must specify the chart's width"):
        shift(
            make_mock_df(),
            score_col=score_col,
            rank_col=rank_col,
            label_col=label_col,
            score_type_col=score_type_col,
            vertical=True,
            width=None,  # Missing width
            autoscale_axes=True,
        )

    # Case 2: autoscale=False + Vertical + No Width -> Pass (because shift_scales is never called)
    try:
        shift(
            make_mock_df(),
            score_col=score_col,
            rank_col=rank_col,
            label_col=label_col,
            score_type_col=score_type_col,
            vertical=True,
            width=None,
            autoscale_axes=False,
        )
    except ValueError:
        pytest.fail("shift() raised ValueError unexpectedly when autoscale_axes=False")


def test_config_propagation(make_mock_df, score_col, rank_col, label_col, score_type_col):
    """
    Check configs passed to the wrapper affect the final chart
    """
    custom_rank_title = "Custom Rank Title"
    custom_score_title = "Custom Score Title"
    custom_bar_colors = {"more_of_positive_counteracted": "magenta"}
    custom_label_color = "cyan"

    rank_config = RankAxisConfig(title=custom_rank_title, grid=False)
    score_config = ScoreAxisConfig(title=custom_score_title, grid=True)
    bar_config = BarConfig(colors=custom_bar_colors, detailed=True)
    label_config = LabelConfig(color=custom_label_color)

    # 2. Generate Chart
    chart = shift(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        label_col=label_col,
        score_type_col=score_type_col,
        width=100,
        vertical=True,
        rank_axis=rank_config,
        score_axis=score_config,
        bar_config=bar_config,
        label_config=label_config,
    )

    layers = chart.to_dict()["layer"]

    # Check bars layer
    bars_encoding = layers[0]["encoding"]

    assert bars_encoding["y"]["axis"]["title"] == custom_rank_title
    assert bars_encoding["y"]["axis"]["grid"] is False

    assert bars_encoding["x"]["axis"]["title"] == custom_score_title
    assert bars_encoding["x"]["axis"]["grid"] is True

    assert "magenta" in bars_encoding["color"]["scale"]["range"]

    # Check labels layer
    label_layer = layers[1]

    assert label_layer["mark"]["color"] == custom_label_color
