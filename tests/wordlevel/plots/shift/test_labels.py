import re
import pytest
import polars as pl
import altair as alt

from polars.testing import assert_frame_equal

from wordlevel.plots.shift.labels import (
    shift_labels,
    get_label_positions,
    to_readable_angle,
    get_position_params,
    to_conditional_expr,
    LabelConfig,
)


def test_get_label_positions(token_col, rank_col, score_col):
    """ """
    df_input = pl.DataFrame(
        {
            token_col: ["a", "a", "b", "b", "c", "c", "d", "d", "e"],
            rank_col: [1, 1, 2, 2, 3, 3, 4, 4, 5],
            score_col: [3, 6, -10, 5, -2, -1, 5, -5, None],
        }
    )

    df_expected = pl.DataFrame(
        {
            token_col: ["a", "b", "c", "d", "e"],
            rank_col: [1, 2, 3, 4, 5],
            score_col: [9.0, -10.0, -3.0, -5.0, 0.0],
        }
    )

    df_actual = get_label_positions(
        df_input, score_col=score_col, rank_col=rank_col, token_col=token_col
    )

    assert_frame_equal(df_actual, df_expected, check_row_order=False, check_column_order=False)


@pytest.mark.parametrize(
    "input_angle, expected_angle",
    [
        (0, 0),  # Right -> Right
        (45, 45),  # Angled -> Angled
        (90, 270),  # Down -> Up (flip 180)
        (100, 280),  # Angled down -> Angled up (flip 180)
        (180, 0),  # Left (upside down) -> Right (flip 180)
        (270, 270),  # Up -> Up
        (-90, -90),  # Up -> Up
        (-45, -45),  # Angled up -> Angled up
        (-150, 30),  # Upside down -> Angled (flip 180)
    ],
)
def test_to_readable_angle(input_angle, expected_angle):
    """ """
    assert to_readable_angle(input_angle) == expected_angle


@pytest.mark.parametrize(
    "bar_angle, rotation, pad, expected_params",
    [
        # Bar and text are congruent: horizontal bar and text (degrees = 0)
        (0, 0, 5, {"align": "left", "baseline": "middle", "dx": 5.0, "dy": 0.0}),
        # Bar and text are congruent: vertical bar and text (degrees = 270)
        # dx is 5 in the rotated space
        (270, 270, 5, {"align": "left", "baseline": "middle", "dx": 5.0, "dy": 0.0}),
        # Bar is vertical (degree = 270) and text is horizontal (degree = 0)
        (270, 0, 5, {"align": "center", "baseline": "bottom", "dx": 0.0, "dy": -5.0}),
        # Bar is horizontal (degree = 0) and text is vertical (degree = 270)
        (0, 270, 5, {"align": "center", "baseline": "top", "dx": 0.0, "dy": 5.0}),
        # Bar is horizontal, text is rotated 45 degrees
        (0, 45, 5, {"align": "left", "baseline": "bottom", "dx": 3.5355, "dy": -3.5355}),
        # Bar is vertical, text is rotated 45 degrees
        (270, 45, 5, {"align": "right", "baseline": "bottom", "dx": -3.5355, "dy": -3.5355}),
    ],
)
def test_get_position_params(bar_angle, rotation, pad, expected_params):
    """ """
    params = get_position_params(bar_angle, rotation, pad)

    assert params["align"] == expected_params["align"]
    assert params["baseline"] == expected_params["baseline"]
    assert pytest.approx(params["dx"], abs=0.001) == expected_params["dx"]
    assert pytest.approx(params["dy"], abs=0.001) == expected_params["dy"]


@pytest.mark.parametrize(
    "key, positive_value, negative_value, expected_if_else",
    [
        # String values should be wrapped in quotes in if/else statement
        ("align", "left", "right", "'left' : 'right'"),
        ("baseline", "top", "bottom", "'top' : 'bottom'"),
        # Numeric values should not be wrapped in quotes in if/else statement
        ("dx", 5.0, -5.0, "5.0 : -5.0"),
        ("dy", 10, -10, "10 : -10"),
    ],
)
def test_to_conditional_expr(score_col, key, positive_value, negative_value, expected_if_else):
    """ """
    pos_params = {key: positive_value}
    neg_params = {key: negative_value}

    expr = to_conditional_expr(score_col, key, pos_params, neg_params)
    assert isinstance(expr, alt.ExprRef)

    expected_expr_str = f"datum['{score_col}'] >= 0 ? {expected_if_else}"
    assert expr.to_dict()["expr"] == expected_expr_str


def test_returns_chart(make_mock_df, score_col, rank_col, token_col):
    """
    Sanity check that the function returns an Altair chart
    """
    chart = shift_labels(
        make_mock_df(), score_col=score_col, rank_col=rank_col, token_col=token_col
    )

    assert isinstance(chart, alt.Chart)


def test_chart_data_and_encodings(make_mock_df, score_col, rank_col, token_col):
    """
    Tests that the chart contains the correct data and the correct encondings are used
    """
    df = make_mock_df(n=50)

    chart = shift_labels(
        df,
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        max_rank=50,
        vertical=True,
    )
    df_chart = pl.DataFrame(chart.data)

    # Check the data is encoded as expected
    df_expected = get_label_positions(
        df, score_col=score_col, rank_col=rank_col, token_col=token_col
    )
    assert_frame_equal(df_chart, df_expected, check_column_order=False, check_row_order=False)

    # Check the chart encoding
    chart_json = chart.to_dict()
    encoding = chart_json["encoding"]

    assert encoding["text"]["field"] == token_col

    assert encoding["x"]["field"] == score_col

    assert encoding["y"]["field"] == rank_col
    assert encoding["y"]["type"] == "ordinal"

    assert "condition" in encoding["angle"]
    assert encoding["angle"]["condition"]["test"] == f"datum['{score_col}'] >= 0"

    mark = chart_json["mark"]
    assert mark["type"] == "text"

    # Check that conditional fields have the conditional pattern
    conditional_pattern = rf"datum\['{re.escape(score_col)}'\]\s*>=\s*0\s*\?\s*.+\s*:\s*.+"

    for conditional_field in ["align", "baseline", "dx", "dy"]:
        assert conditional_field in mark
        assert "expr" in mark[conditional_field]
        assert re.search(conditional_pattern, mark[conditional_field]["expr"])


@pytest.mark.parametrize("vertical", [True, False])
def test_chart_orientation(make_mock_df, score_col, rank_col, token_col, vertical):
    """
    Test that the vertical parameter correctly swaps the X and Y axes.
    """
    chart = shift_labels(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
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


@pytest.mark.parametrize("filter_max_rank", [True, False])
def test_max_rank_filter(make_mock_df, score_col, rank_col, token_col, filter_max_rank):
    """
    Test that max_rank correctly filters the data used for the chart
    """
    n = 50
    max_rank = 11

    chart = shift_labels(
        make_mock_df(n=n),
        rank_col=rank_col,
        score_col=score_col,
        token_col=token_col,
        max_rank=max_rank,
        filter_max_rank=filter_max_rank,
    )

    df_chart = pl.DataFrame(chart.data)

    if filter_max_rank:
        n_expected = max_rank
    else:
        n_expected = n

    assert df_chart["rank"].max() == n_expected


@pytest.mark.parametrize(
    "color, size, weight",
    [
        ("red", 10, "bold"),
        ("blue", 100, 100),
    ],
)
def test_label_config(make_mock_df, score_col, rank_col, token_col, color, size, weight):
    """ """
    # Test that the config works whether given a formal config or a dictionary
    for constructor in [LabelConfig, dict]:
        config_params = {"color": color, "size": size, "weight": weight}
        config = constructor(**config_params)

        config = LabelConfig(color=color, size=size, weight=weight)
        df = make_mock_df()

        chart = shift_labels(
            df, score_col=score_col, rank_col=rank_col, token_col=token_col, config=config
        )
        chart_json = chart.to_dict()

        assert chart_json["mark"]["color"] == color
        assert chart_json["mark"]["fontSize"] == size
        assert chart_json["mark"]["fontWeight"] == weight


def test_default_label_config(make_mock_df, score_col, rank_col, token_col):
    """ """
    default_config = LabelConfig()
    df = make_mock_df()

    chart = shift_labels(df, score_col=score_col, rank_col=rank_col, token_col=token_col)
    chart_json = chart.to_dict()

    assert chart_json["mark"]["color"] == default_config.color
    assert chart_json["mark"]["fontWeight"] == default_config.weight
    assert chart_json["mark"]["fontSize"] == default_config.size


@pytest.mark.parametrize(
    "on_bar_ends, vertical",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_label_config_on_bar_ends(
    make_mock_df, score_col, rank_col, token_col, on_bar_ends, vertical
):
    """ """
    chart = shift_labels(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        vertical=vertical,
        config=LabelConfig(on_bar_ends=on_bar_ends),
    )

    chart_json = chart.to_dict()

    if vertical:
        axis = chart_json["encoding"]["x"]
    else:
        axis = chart_json["encoding"]["y"]

    if on_bar_ends:
        # If True, the labels should be based on the token positions (specified by score_col)
        assert "field" in axis
        assert axis["field"] == score_col
    else:
        # If False, the labels should sit at a static position (datum)
        assert "datum" in axis
        assert isinstance(axis["datum"], (int, float))


@pytest.mark.parametrize(
    "mirror_rotation, rotation", [(True, 20), (True, -50), (False, 72), (False, -33)]
)
def test_label_config_mirror_rotation(
    make_mock_df, score_col, rank_col, token_col, mirror_rotation, rotation
):
    """ """
    chart = shift_labels(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        config=LabelConfig(rotation=rotation, mirror_rotation=mirror_rotation),
    )

    chart_json = chart.to_dict()["encoding"]
    angle = chart_json["angle"]

    expected_pos_text_angle = to_readable_angle(rotation)
    if mirror_rotation:
        # If mirrored, negative bars use the inverse (-45 -> readable)
        expected_neg_text_angle = to_readable_angle(-1 * rotation)
    else:
        # If not mirrored, negative bars use the same angle as positive bars
        expected_neg_text_angle = expected_pos_text_angle

    assert angle["condition"]["value"] == expected_pos_text_angle
    assert angle["value"] == expected_neg_text_angle
