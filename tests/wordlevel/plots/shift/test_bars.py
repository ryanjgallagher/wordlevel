import altair as alt
import polars as pl
import pytest

from polars.testing import assert_frame_equal

from wordlevel.configs import ShiftTypesConfig
from wordlevel.plots.shift.bars import (
    shift_bars,
    RankAxisConfig,
    ScoreAxisConfig,
    InternalBarConfig,
    InternalScoreAxisConfig,
)


def test_returns_chart(make_mock_df, score_col, score_type_col, rank_col):
    """
    Sanity check that the function returns an Altair chart
    """
    chart = shift_bars(
        make_mock_df(), rank_col=rank_col, score_col=score_col, score_type_col=score_type_col
    )

    assert isinstance(chart, alt.Chart)


def test_chart_data_and_encodings(make_mock_df, score_col, score_type_col, rank_col):
    """
    Tests that the chart contains the correct data and the correct encondings are used
    """
    df = make_mock_df(n=50)

    chart = shift_bars(
        df,
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        max_rank=50,
        vertical=True,
    )

    df_chart = pl.DataFrame(chart.data)
    assert_frame_equal(df, df_chart)

    chart_json = chart.to_dict()
    encoding = chart_json["encoding"]

    assert encoding["x"]["field"] == score_col
    assert encoding["x"]["type"] == "quantitative"

    assert encoding["y"]["field"] == rank_col
    assert encoding["y"]["type"] == "ordinal"

    assert encoding["color"]["field"] == score_type_col
    assert encoding["color"]["type"] == "nominal"


@pytest.mark.parametrize("vertical", [True, False])
def test_chart_orientation(make_mock_df, score_col, score_type_col, rank_col, vertical):
    """
    Test that the vertical parameter correctly swaps the X and Y axes.
    """
    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=vertical,
    )

    chart_json = chart.to_dict()
    encoding = chart_json["encoding"]

    if vertical:
        assert encoding["x"]["field"] == score_col
        assert encoding["y"]["field"] == rank_col
    else:
        assert encoding["x"]["field"] == rank_col
        assert encoding["y"]["field"] == score_col


@pytest.mark.parametrize("width, height", [(800, 600), (None, 500), (300, None), (None, None)])
def test_chart_dimensions(make_mock_df, score_col, score_type_col, rank_col, width, height):
    """
    Test that width and height are specified correctly
    """
    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        width=width,
        height=height,
    )

    chart_json = chart.to_dict()

    if width is not None:
        assert chart_json["width"] == width
    else:
        assert "width" not in chart_json

    if height is not None:
        assert chart_json["height"] == height
    else:
        assert "height" not in chart_json


@pytest.mark.parametrize("filter_max_rank", [True, False])
def test_max_rank_filter(make_mock_df, score_col, score_type_col, rank_col, filter_max_rank):
    """
    Test that max_rank correctly filters the data used for the chart
    """
    n = 50
    max_rank = 11

    chart = shift_bars(
        make_mock_df(n=n),
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        max_rank=max_rank,
        filter_max_rank=filter_max_rank,
    )

    df_chart = pl.DataFrame(chart.data)

    if filter_max_rank:
        n_expected = max_rank
    else:
        n_expected = n

    assert df_chart["rank"].max() == n_expected
    assert df_chart.height == n_expected * len(ShiftTypesConfig().colors())


@pytest.mark.parametrize(
    "Config,title,title_fontsize,label_fontsize,grid",
    [
        (RankAxisConfig, None, None, None, True),
        (RankAxisConfig, "My custom title", 44, 22, False),
        (InternalScoreAxisConfig, None, None, None, False),
        (InternalScoreAxisConfig, "Another title", 33, 11, True),
    ],
)
def test_axis_config(
    make_mock_df,
    score_col,
    score_type_col,
    rank_col,
    Config,
    title,
    title_fontsize,
    label_fontsize,
    grid,
):
    """
    Test that configs for the rank axis are properly applied to the chart
    """
    # Test that the config works whether given a formal config or a dictionary
    for constructor in [Config, dict]:
        config_params = {
            "title": title,
            "title_fontsize": title_fontsize,
            "label_fontsize": label_fontsize,
            "grid": grid,
        }
        config = constructor(**config_params)

        chart_params = {
            "df": make_mock_df(),
            "rank_col": rank_col,
            "score_col": score_col,
            "score_type_col": score_type_col,
            "vertical": True,
        }
        if isinstance(config, RankAxisConfig):
            chart = shift_bars(**chart_params, rank_axis=config)
            axis = "y"
        elif isinstance(config, ScoreAxisConfig):
            chart = shift_bars(**chart_params, score_axis=config)
            axis = "x"

        chart_json = chart.to_dict()
        chart_axis = chart_json["encoding"][axis]["axis"]

        altair_key2expected_param = {
            "title": title,
            "titleFontSize": title_fontsize,
            "labelFontSize": label_fontsize,
        }

        for altair_key, expected_param in altair_key2expected_param.items():
            if expected_param is None:
                assert altair_key not in chart_axis or chart_axis[altair_key] is None
            else:
                assert chart_axis[altair_key] == expected_param

        if grid:
            assert chart_axis["grid"]
        else:
            assert not chart_axis["grid"]


def test_axis_default_configs(make_mock_df, score_col, score_type_col, rank_col):
    """
    Tests that the default configs work as expected
    """
    rank_config = RankAxisConfig()
    score_config = ScoreAxisConfig()

    chart = shift_bars(
        make_mock_df(),
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        vertical=True,
        rank_axis=None,
        score_axis=None,
    )
    chart_json = chart.to_dict()

    rank_axis = chart_json["encoding"]["y"]["axis"]
    assert rank_axis["title"] == rank_config.title
    assert rank_axis["grid"] == rank_config.grid

    score_axis = chart_json["encoding"]["x"]["axis"]
    assert score_axis["title"] == score_config.title
    assert score_axis["grid"] == score_config.grid


@pytest.mark.parametrize("all_positive_labels", [True, False])
def test_score_axis_all_positive_labels(
    make_mock_df, score_col, score_type_col, rank_col, all_positive_labels
):
    """
    Test that all_positive_labels correctly adds a labelExpr to show absolute values on the
    score axis. Since Vega evaluates labelExpr at render time, we verify the expression in the
    JSON spec rather than the rendered token strings.
    """
    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=True,
        score_axis=InternalScoreAxisConfig(all_positive_labels=all_positive_labels),
    )

    chart_json = chart.to_dict()
    score_axis = chart_json["encoding"]["x"]["axis"]

    if all_positive_labels:
        assert score_axis["labelExpr"] == "abs(datum.value)"
    else:
        assert "labelExpr" not in score_axis


@pytest.mark.parametrize(
    "tick_values, tick_step, first_tick, max_rank, expected_output",
    [
        # Explicit values are passed. Even though step=5 is set, the ticks should be [10, 20]
        ([10, 20], 5, True, 50, [10, 20]),
        # Defaults are passed, with a lower max_rank
        (None, 5, True, 13, [1, 5, 10]),
        # No first tick
        (None, 5, False, 13, [5, 10]),
        # Step size = 1. Check that function does not double apply `1` to ticks, i.e. [1, 1, 2, 3]
        (None, 1, True, 3, [1, 2, 3]),
        # Step size > max rank
        (None, 50, True, 10, [1]),
        # Altair defaults
        (None, None, True, 50, None),
    ],
)
def test_rank_axis_tick_configs(
    make_mock_df,
    score_col,
    score_type_col,
    rank_col,
    tick_values,
    tick_step,
    first_tick,
    max_rank,
    expected_output,
):
    """
    Tests that rank axis tick configurations can be specified correctly
    """
    chart = shift_bars(
        make_mock_df(),
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        max_rank=max_rank,
        vertical=True,
        rank_axis=RankAxisConfig(
            label_tick_values=tick_values, label_tick_step=tick_step, label_first_tick=first_tick
        ),
    )

    chart_json = chart.to_dict()
    axis = chart_json["encoding"]["y"]["axis"]

    if expected_output is None:
        assert "values" not in axis
    else:
        assert axis["values"] == expected_output


@pytest.mark.parametrize(
    "colors",
    [
        (None),
        ("category"),
        (
            {
                "lower_score": "#d73027",
                "lower_score_counteracted": "#fce7e5",
                "more_of_negative": "#fc8d59",
                "more_of_negative_counteracted": "#fee090",
                "less_of_positive": "#a50026",
                "less_of_positive_counteracted": "#d73027",
                "higher_score": "#4575b4",
                "higher_score_counteracted": "#e0f3f8",
                "more_of_positive": "#91bfdb",
                "more_of_positive_counteracted": "#eef6fa",
                "less_of_negative": "#313695",
                "less_of_negative_counteracted": "#aebcd9",
            }
        ),
    ],
)
def test_bar_configs_not_detailed(make_mock_df, score_col, score_type_col, rank_col, colors):
    """
    Tests non-detailed configurations of bars, including the default, passing a string, and passing
    a custom color dictionary
    """
    default_config = InternalBarConfig()
    df = make_mock_df()
    expected_domain = set(df[score_type_col].unique())

    chart_params = {
        "df": df,
        "score_col": score_col,
        "score_type_col": score_type_col,
        "rank_col": rank_col,
        "bars": None,
    }
    if colors is not None:
        chart_params["bars"] = InternalBarConfig(colors=colors)

    chart = shift_bars(**chart_params)

    chart_json = chart.to_dict()
    scale = chart_json["encoding"]["color"]["scale"]

    assert set(scale["domain"]) == expected_domain

    if colors is None:
        assert scale["range"] == default_config.colors
    elif isinstance(colors, str):
        assert scale["range"] == colors
    elif isinstance(colors, dict):
        assert set(scale["range"]) == set(colors.values())


def test_bar_configs_incomplete_colors(make_mock_df, score_col, score_type_col, rank_col):
    """
    Test that the bar configuration fails if a user does not provide a complete color mapping
    """
    with pytest.raises(KeyError):
        _ = shift_bars(
            make_mock_df(),
            rank_col=rank_col,
            score_col=score_col,
            score_type_col=score_type_col,
            bars=InternalBarConfig(colors={"lower_score": "red"}, detailed=False),
        )


def test_bar_configs_detailed(make_mock_df, score_col, score_type_col, rank_col):
    """
    Test that detailed bar configurations work as expected
    """
    chart = shift_bars(
        make_mock_df(),
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        bars=InternalBarConfig(detailed=True),
    )

    chart_json = chart.to_dict()
    scale = chart_json["encoding"]["color"]["scale"]

    default_colors = ShiftTypesConfig().colors()

    assert set(scale["domain"]) == set(default_colors.keys())
    assert set(scale["range"]) == set(default_colors.values())

    color_mapping = dict(zip(scale["domain"], scale["range"]))
    for score_type, expected_color in default_colors.items():
        assert color_mapping[score_type] == expected_color

    assert chart_json["encoding"]["order"]["field"] == "stacked_bar_order"


def test_bar_configs_detailed_custom_colors(make_mock_df, score_col, score_type_col, rank_col):
    """
    Test that detailed bar configurations overrides custom colors while maintaining default colors
    """
    custom_colors = {"lower_score": "#000000", "more_of_positive": "#999999"}

    chart = shift_bars(
        make_mock_df(),
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        bars=InternalBarConfig(detailed=True, colors=custom_colors),
    )

    chart_json = chart.to_dict()
    scale = chart_json["encoding"]["color"]["scale"]

    default_colors = ShiftTypesConfig().colors()
    color_mapping = dict(zip(scale["domain"], scale["range"]))
    for score_type, default_color in default_colors.items():
        if score_type in custom_colors:
            assert color_mapping[score_type] == custom_colors[score_type]
        else:
            assert color_mapping[score_type] == default_color


def test_bar_configs_custom_orders(make_mock_df, score_col, score_type_col, rank_col):
    """
    Test that the bar configurations properly execute custom bar orders
    """
    custom_orders = {"lower_score": 22, "higher_score": 7}
    df = make_mock_df().filter(pl.col(score_type_col).is_in(custom_orders))

    chart = shift_bars(
        df,
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        bars=InternalBarConfig(orders=custom_orders),
    )

    chart_json = chart.to_dict()

    assert chart_json["encoding"]["order"]["field"] == "stacked_bar_order"

    df_chart = pl.DataFrame(chart.data)
    assert "stacked_bar_order" in df_chart.columns

    for score_type, expected_order in custom_orders.items():
        order = (
            df_chart.filter(pl.col(score_type_col) == score_type)
            .select("stacked_bar_order")
            .unique()
            .item()
        )
        assert int(order) == expected_order
