import altair as alt
import polars as pl
import pytest

from polars.testing import assert_frame_equal

from wordlevel.plots.shift.labels import LabelConfig
from wordlevel.plots.shift.scales import shift_scales, get_scale_limits, get_bar_lengths


def test_get_bar_lengths(score_col, rank_col, label_col):
    """ """
    df_input = pl.DataFrame(
        {
            label_col: ["w1", "w1", "w1", "w2", "w2", "w2", "w3", "w3", "w3", "w4"],
            rank_col: [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
            score_col: [5, -3, 4, -6, -2, 7, 1, 0, -1, 7],
        }
    )

    df_expected = pl.DataFrame(
        {label_col: ["w1", "w2", "w3", "w4"], rank_col: [1, 2, 3, 4], "bar_length": [9, 8, 1, 7]}
    )

    df_actual = get_bar_lengths(
        df_input, score_col=score_col, rank_col=rank_col, label_col=label_col
    )

    assert_frame_equal(
        df_actual.select(df_expected.columns),
        df_expected,
        check_row_order=False,
        check_column_order=False,
        check_dtypes=False,
    )


@pytest.mark.parametrize(
    "score,size,fontsize,font_width_height_ratio,scale_factor,pad,on_bar_ends,expected_limit",
    [
        # fmt: off
        (5, 1, 1, 1, 1, 0, True, 25),  # Base case
        (5, 1, 1, 1, 1, 0, False, 20),  # Not on bar ends
        (5, 1, 1, 1, 1, 3, True, 55),  # Padding
        (5, 0.5, 1, 1, 1, 0, True, 45),  # Alter size (smaller)
        (5, 5, 1, 1, 1, 0, True, 9),  # Alter size (larger)
        (5, 1, 2, 1, 1, 0, True, 45),  # Alter fontsize (larger)
        (5, 1, 0.5, 1, 1, 0, True, 15),  # Alter fontsize (smaller)
        (5, 1, 1, 2, 1, 0, True, 45),  # Alter font width:height ratio (larger)
        (5, 1, 1, 0.5, 1, 0, True, 15),  # Alter font width:height ratio (smaller)
        (5, 1, 2, 0.5, 1, 0, True, 25),  # Alter fontsize and font width:height ratio together
        (5, 1, 1, 1, 4, 0, True, 100),  # Alter scale factor (larger)
        (5, 1, 1, 1, 0.2, 0, True, 5),  # Alter scale factor (smaller)
        # fmt: on
    ],
)
def test_get_scale_limits(
    score_col,
    rank_col,
    label_col,
    score,
    size,
    fontsize,
    font_width_height_ratio,
    scale_factor,
    pad,
    on_bar_ends,
    expected_limit,
):
    """ """
    df_input = pl.DataFrame(
        {
            label_col: ["w1", "w2", "w3"],
            rank_col: [1, 2, 3],
            score_col: [score, score - 1, score - 10],
        }
    )

    df_expected = pl.DataFrame(
        {rank_col: [1], "positive_limit": expected_limit, "negative_limit": -1 * expected_limit}
    )

    df_actual = get_scale_limits(
        df_input,
        score_col=score_col,
        rank_col=rank_col,
        label_col=label_col,
        size=size,
        fontsize=fontsize,
        font_width_to_height_ratio=font_width_height_ratio,
        scale_factor=scale_factor,
        pad=pad,
        on_bar_ends=on_bar_ends,
    )

    assert_frame_equal(df_actual, df_expected, check_dtypes=False)


def test_get_scale_limits_all_positive(
    score_col,
    rank_col,
    label_col,
):
    df_input = pl.DataFrame(
        {
            label_col: ["w1", "w2", "w3"],
            rank_col: [1, 2, 3],
            score_col: [5, 4, 3],
        }
    )

    df_expected = pl.DataFrame({rank_col: [1], "positive_limit": 25, "negative_limit": 0})

    df_actual = get_scale_limits(
        df_input,
        score_col=score_col,
        rank_col=rank_col,
        label_col=label_col,
        size=1,
        fontsize=1,
        font_width_to_height_ratio=1,
        scale_factor=1,
        pad=0,
        on_bar_ends=True,
    )

    assert_frame_equal(df_actual, df_expected, check_dtypes=False)


def test_returns_chart(make_mock_df, score_col, rank_col, label_col):
    """
    Sanity check that the function returns an Altair chart
    """
    chart = shift_scales(
        make_mock_df(), score_col=score_col, rank_col=rank_col, label_col=label_col, width=10
    )

    assert isinstance(chart, alt.LayerChart)


@pytest.mark.parametrize("vertical, wrong_size_col", [(True, "height"), (False, "width")])
def test_error_if_size_not_specified(
    make_mock_df, score_col, rank_col, label_col, vertical, wrong_size_col
):
    """ """
    with pytest.raises(ValueError):
        _ = shift_scales(
            make_mock_df(),
            score_col=score_col,
            rank_col=rank_col,
            label_col=label_col,
            vertical=vertical,
            **{wrong_size_col: 10},
        )


@pytest.mark.parametrize("vertical", [True, False])
def test_chart_orientation(make_mock_df, score_col, rank_col, label_col, vertical):
    """
    Test that the vertical parameter correctly swaps the X and Y axes
    for the scale layers.
    """
    chart = shift_scales(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        label_col=label_col,
        vertical=vertical,
        width=100,
        height=100,
    )
    chart_json = chart.to_dict()

    for layer, axis in zip(chart_json["layer"], ["positive_limit", "negative_limit"]):
        encoding = layer["encoding"]

        if vertical:
            assert encoding["x"]["field"] == axis
            assert encoding["y"]["value"] == 0
        else:
            assert encoding["x"]["value"] == 0
            assert encoding["y"]["field"] == axis


@pytest.mark.parametrize("filter_max_rank", [True, False])
def test_max_rank_filter(score_col, rank_col, label_col, filter_max_rank):
    """
    Test that max_rank correctly filters the data used for the chart
    """
    df_mock = pl.DataFrame({label_col: ["w1", "w2"], rank_col: [100, 1], score_col: [100, 5]})

    if filter_max_rank:
        expected_limit = 25
    else:
        expected_limit = 500

    chart = shift_scales(
        df_mock,
        score_col=score_col,
        rank_col=rank_col,
        label_col=label_col,
        max_rank=2,
        filter_max_rank=filter_max_rank,
        width=1,
        scale_factor=1,
        # Test passing a dictionary config works vs passing a formal config
        config={"size": 1, "font_width_to_height_ratio": 1, "pad": 0},
    )

    # chart_json['datasets'] is of the form {arbitrary_key: [{rank: 1, pos_lim: N, neg_lim: -N}]}
    chart_data = list(chart.to_dict()["datasets"].values())[0]

    assert len(chart_data) == 1
    assert chart_data[0]["positive_limit"] == expected_limit


def test_default_fontsize(make_mock_df, score_col, rank_col, label_col):
    """ """
    chart_config = shift_scales(
        make_mock_df(), score_col, rank_col, label_col, width=10, config=LabelConfig()
    )
    chart_none = shift_scales(
        make_mock_df(), score_col, rank_col, label_col, width=10, config=LabelConfig(size=None)
    )

    data_config = list(chart_config.to_dict()["datasets"].values())[0]
    data_none = list(chart_none.to_dict()["datasets"].values())[0]
    limit_config = data_config[0]["positive_limit"]
    limit_none = data_none[0]["positive_limit"]

    # Limits should be the same if the chart defaulted correctly
    assert limit_config == limit_none


def test_layers_are_invisible(make_mock_df, score_col, rank_col, label_col):
    """
    Tests that the scale layers are invisible (opacity 0) and have tooltips disabled.
    """
    chart = shift_scales(make_mock_df(), score_col, rank_col, label_col, width=10)

    layers = chart.to_dict()["layer"]

    for layer in layers:
        mark = layer["mark"]
        assert isinstance(mark, dict)
        assert mark["opacity"] == 0

        encoding = layer["encoding"]
        assert "tooltip" in encoding
        assert encoding["tooltip"]["value"] is None
