import altair as alt
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from wordlevel.configs import (
    BAR_OUTLINE_FIELDS,
    DEFAULT_BAR_OUTLINE_COLOR,
    DEFAULT_BAR_OUTLINE_WIDTH,
    DEFAULT_SCORE_AXIS_TITLE,
    PERCENT_SCORE_AXIS_TITLE,
    BarConfig,
    RankAxisConfig,
    ScoreAxisConfig,
    ScoreSignBarConfig,
    ShiftBarConfig,
    TotalShiftConfig,
)
from wordlevel.constants import SHIFT_SCORE_TYPES
from wordlevel.plots.core.shift.bars import (
    _SHIFT_TYPE_ORDERS,
    InternalBarConfig,
    InternalScoreAxisConfig,
    shift_bars,
)
from wordlevel.plots.core.shift.border import InternalBorderConfig


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
    Tests that the chart contains the correct data and the correct encodings are used
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


@pytest.mark.parametrize("vertical", [True, False])
@pytest.mark.parametrize("bar_width", [10, 14, 25])
def test_rank_dimension_from_bar_width(
    make_mock_df, score_col, score_type_col, rank_col, vertical, bar_width
):
    """
    The rank-axis dimension (height when vertical, width when horizontal) is an alt.Step derived
    from RankAxisConfig.bar_width, so each rank occupies a fixed size regardless of bar count.
    """
    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=vertical,
        rank_axis=RankAxisConfig(bar_width=bar_width),
    )

    spec = chart.to_dict()
    rank_dim = "height" if vertical else "width"
    assert spec[rank_dim] == {"step": bar_width}


@pytest.mark.parametrize("vertical", [True, False])
def test_rank_dimension_ignores_explicit_size(
    make_mock_df, score_col, score_type_col, rank_col, vertical
):
    """
    The rank-axis dimension always comes from bar_width, so an explicit pixel size passed for that
    axis (height when vertical, width when horizontal) is ignored. The score-axis dimension in the
    other direction is still honored.
    """
    bar_width = RankAxisConfig().bar_width

    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=vertical,
        height=999,
        width=777,
    )

    spec = chart.to_dict()
    if vertical:
        assert spec["height"] == {"step": bar_width}  # explicit height ignored
        assert spec["width"] == 777  # score-axis width honored
    else:
        assert spec["width"] == {"step": bar_width}  # explicit width ignored
        assert spec["height"] == 999  # score-axis height honored


@pytest.mark.parametrize("vertical", [True, False])
def test_score_dimension_absent_when_unset(
    make_mock_df, score_col, score_type_col, rank_col, vertical
):
    """
    When the score-axis dimension is not given it is left out of the spec, while the rank-axis
    dimension is still the bar_width Step.
    """
    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=vertical,
    )

    spec = chart.to_dict()
    bar_width = RankAxisConfig().bar_width
    if vertical:
        assert spec["height"] == {"step": bar_width}
        assert "width" not in spec
    else:
        assert spec["width"] == {"step": bar_width}
        assert "height" not in spec


@pytest.mark.parametrize(
    "border, expected_color",
    [
        # The default internal border is Altair's axis grey
        (None, "#888"),
        (InternalBorderConfig(), "#888"),
        # Explicitly opting out (color=None) defers to Altair's per-element defaults (keys absent)
        (InternalBorderConfig(color=None), None),
        (InternalBorderConfig(color="black", width=2), "black"),
    ],
)
def test_border_config(make_mock_df, score_col, score_type_col, rank_col, border, expected_color):
    """
    shift_bars draws both the rank and score axis domains, colored by the border color (the fallback
    for axes without their own axis_color). There is no view box; the opposite border edges are drawn
    separately by shift_borders. The border color is also the fallback tick color on both axes.
    """
    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=True,
        border=border,
    )

    spec = chart.to_dict()

    # No view box is drawn — borders are axis-domain lines
    assert "stroke" not in spec.get("view", {})

    # Both the rank (y) and score (x) axes draw their own domain, colored by the border color
    for axis_channel in ("x", "y"):
        axis = spec["encoding"][axis_channel]["axis"]
        assert axis["domain"] is True
        if expected_color is None:
            assert "domainColor" not in axis
            assert "tickColor" not in axis
        else:
            assert axis["domainColor"] == expected_color
            assert axis["tickColor"] == expected_color


@pytest.mark.parametrize(
    "axis_config_cls, channel",
    [(RankAxisConfig, "y"), (InternalScoreAxisConfig, "x")],
)
def test_axis_width_owned_by_axis_config(
    make_mock_df, score_col, score_type_col, rank_col, axis_config_cls, channel
):
    """
    The native axis domain width is owned by the axis config's own `width`; it does not fall back to
    `BorderConfig.width`, which governs only the opposite border edges.
    """
    config = axis_config_cls(width=3)
    kwargs = {"rank_axis": config} if channel == "y" else {"score_axis": config}

    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=True,
        border=InternalBorderConfig(width=1),  # differs from the axis width; must not leak in
        **kwargs,
    )

    axis = chart.to_dict()["encoding"][channel]["axis"]
    assert axis["domainWidth"] == 3


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
    assert df_chart.height == n_expected * len(SHIFT_SCORE_TYPES)


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
    # The parametrized `Config` selects the rank or score axis; exercise it both as a formal config
    # object and as a plain dict of the same fields.
    is_rank = issubclass(Config, RankAxisConfig)
    axis = "y" if is_rank else "x"

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
        axis_kwargs = {"rank_axis": config} if is_rank else {"score_axis": config}
        chart = shift_bars(**chart_params, **axis_kwargs)

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


@pytest.mark.parametrize(
    "axis_config_cls, channel", [(RankAxisConfig, "y"), (InternalScoreAxisConfig, "x")]
)
def test_axis_text_and_grid_colors(
    make_mock_df, score_col, score_type_col, rank_col, axis_config_cls, channel
):
    """
    label_color, title_color, and grid_color on the rank/score axis configs pass through to the
    corresponding alt.Axis keys. These are the per-axis text/gridline colors, distinct from the
    structural stroke color set by BorderConfig (domainColor/tickColor).
    """
    config = axis_config_cls(label_color="#111111", title_color="#222222", grid_color="#333333")

    kwargs = {"rank_axis": config} if channel == "y" else {"score_axis": config}
    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=True,
        **kwargs,
    )

    axis = chart.to_dict()["encoding"][channel]["axis"]
    assert axis["labelColor"] == "#111111"
    assert axis["titleColor"] == "#222222"
    assert axis["gridColor"] == "#333333"


@pytest.mark.parametrize(
    "axis_config_cls, channel", [(RankAxisConfig, "y"), (InternalScoreAxisConfig, "x")]
)
def test_axis_colors_absent_by_default(
    make_mock_df, score_col, score_type_col, rank_col, axis_config_cls, channel
):
    """
    When the per-axis colors are left at their default (None), the corresponding keys are not
    emitted, so Altair falls back to its own defaults.
    """
    kwargs = (
        {"rank_axis": axis_config_cls()} if channel == "y" else {"score_axis": axis_config_cls()}
    )
    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=True,
        **kwargs,
    )

    axis = chart.to_dict()["encoding"][channel]["axis"]
    for key in ("labelColor", "titleColor", "gridColor"):
        assert key not in axis


@pytest.mark.parametrize(
    "axis_config_cls, channel", [(RankAxisConfig, "y"), (InternalScoreAxisConfig, "x")]
)
@pytest.mark.parametrize(
    "axis_color, tick_color, border_color, expected_domain, expected_tick",
    [
        # Neither set -> both fall back to the border color
        (None, None, "black", "black", "black"),
        # axis_color overrides the line; tick_color independently falls back to border
        ("red", None, "black", "red", "black"),
        # tick_color overrides the ticks; axis_color independently falls back to border
        (None, "blue", "black", "black", "blue"),
        # Both override the border
        ("red", "blue", "black", "red", "blue"),
        # Nothing set anywhere -> keys absent (Altair default)
        (None, None, None, None, None),
        # axis_color works even without a border color
        ("red", None, None, "red", None),
    ],
)
def test_axis_color_overrides_and_fallback(
    make_mock_df,
    score_col,
    score_type_col,
    rank_col,
    axis_config_cls,
    channel,
    axis_color,
    tick_color,
    border_color,
    expected_domain,
    expected_tick,
):
    """
    `axis_color`/`tick_color` on an axis config override the BorderConfig color independently of
    each other; when either is None it falls back to `border.color` (and is omitted entirely when
    that is None too).
    """
    config = axis_config_cls(axis_color=axis_color, tick_color=tick_color)
    kwargs = {"rank_axis": config} if channel == "y" else {"score_axis": config}

    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=True,
        border=InternalBorderConfig(color=border_color),
        **kwargs,
    )

    axis = chart.to_dict()["encoding"][channel]["axis"]

    # Both the rank and score axes draw their domain, so domain and tick colors behave identically
    for key, expected in (("domainColor", expected_domain), ("tickColor", expected_tick)):
        if expected is None:
            assert key not in axis
        else:
            assert axis[key] == expected


def test_axis_colors_are_per_axis(make_mock_df, score_col, score_type_col, rank_col):
    """
    The rank-axis and score-axis line/tick colors are independent of one another (setting one does
    not leak into the other).
    """
    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=True,
        rank_axis=RankAxisConfig(axis_color="red", tick_color="green"),
        score_axis=InternalScoreAxisConfig(axis_color="blue", tick_color="orange"),
        border=InternalBorderConfig(color="black"),
    )

    encoding = chart.to_dict()["encoding"]
    rank_axis = encoding["y"]["axis"]  # rank axis is y when vertical
    score_axis = encoding["x"]["axis"]

    # Both axes draw their own domain and ticks, independently colored
    assert rank_axis["domainColor"] == "red"
    assert rank_axis["tickColor"] == "green"
    assert score_axis["domainColor"] == "blue"
    assert score_axis["tickColor"] == "orange"


@pytest.mark.parametrize("all_positive_labels", [True, False])
def test_score_axis_all_positive_labels(
    make_mock_df, score_col, score_type_col, rank_col, all_positive_labels
):
    """
    Test that all_positive_labels correctly adds a labelExpr to show absolute values on the
    score axis. Since Vega evaluates labelExpr at render time, we verify the expression in the
    JSON spec rather than the rendered label strings.
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
    "percent, all_positive_labels, expected_label_expr",
    [
        (False, False, None),
        (False, True, "abs(datum.value)"),
        (True, False, "datum.value + '%'"),
        (True, True, "abs(datum.value) + '%'"),
    ],
)
def test_score_axis_percent_labels(
    make_mock_df,
    score_col,
    score_type_col,
    rank_col,
    percent,
    all_positive_labels,
    expected_label_expr,
):
    """When `percent` is set, the score-axis tick labels are suffixed with a percent sign."""
    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=True,
        score_axis=InternalScoreAxisConfig(
            all_positive_labels=all_positive_labels, percent=percent
        ),
    )

    score_axis = chart.to_dict()["encoding"]["x"]["axis"]

    if expected_label_expr is None:
        assert "labelExpr" not in score_axis
    else:
        assert score_axis["labelExpr"] == expected_label_expr


@pytest.mark.parametrize(
    "config, percent, expected_title",
    [
        # The default title becomes the percent title only when shown as percentages
        (None, False, DEFAULT_SCORE_AXIS_TITLE),
        (None, True, PERCENT_SCORE_AXIS_TITLE),
        # A custom title is preserved, even when shown as percentages
        (ScoreAxisConfig(title="Custom"), True, "Custom"),
        # An explicitly hidden title (None) stays hidden (no title rendered)
        (ScoreAxisConfig(title=None), True, None),
    ],
)
def test_score_axis_title_resolved_and_rendered(
    make_mock_df, score_col, score_type_col, rank_col, config, percent, expected_title
):
    """The resolved title (percent only swaps the default) is carried through to the chart."""
    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=True,
        score_axis=InternalScoreAxisConfig.from_config(
            config, all_positive_labels=False, percent=percent
        ),
    )

    score_axis = chart.to_dict()["encoding"]["x"]["axis"]
    # Indexing (not `.get`) so a missing title, which Vega-Lite renders as the field name, fails
    assert score_axis["title"] == expected_title


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


def test_shift_bar_configs_default(make_mock_df, score_col, score_type_col, rank_col):
    """
    Test that the default shift bar configuration applies the ShiftBarConfig defaults
    """
    chart = shift_bars(
        make_mock_df(),
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        bars=InternalBarConfig.from_shift_bar_config(None),
    )

    chart_json = chart.to_dict()
    scale = chart_json["encoding"]["color"]["scale"]

    default_colors = {
        score_type: getattr(ShiftBarConfig(), score_type) for score_type in SHIFT_SCORE_TYPES
    }

    assert set(scale["domain"]) == set(default_colors.keys())
    assert set(scale["range"]) == set(default_colors.values())

    color_mapping = dict(zip(scale["domain"], scale["range"]))
    for score_type, expected_color in default_colors.items():
        assert color_mapping[score_type] == expected_color

    assert chart_json["encoding"]["order"]["field"] == "stacked_bar_order"


@pytest.mark.parametrize("as_dict", [False, True], ids=["object", "dict"])
def test_shift_bar_configs_custom_colors(
    make_mock_df, score_col, score_type_col, rank_col, as_dict
):
    """
    Test that a custom ShiftBarConfig overrides individual colors while keeping the rest at default,
    whether passed as a ShiftBarConfig object or as a dict of the same fields
    """
    custom_overrides = {"lower_score": "#000000", "more_of_positive": "#999999"}
    config = custom_overrides if as_dict else ShiftBarConfig(**custom_overrides)

    chart = shift_bars(
        make_mock_df(),
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        bars=InternalBarConfig.from_shift_bar_config(config),
    )

    chart_json = chart.to_dict()
    scale = chart_json["encoding"]["color"]["scale"]

    default_colors = {
        score_type: getattr(ShiftBarConfig(), score_type) for score_type in SHIFT_SCORE_TYPES
    }
    color_mapping = dict(zip(scale["domain"], scale["range"]))
    for score_type, default_color in default_colors.items():
        if score_type in custom_overrides:
            assert color_mapping[score_type] == custom_overrides[score_type]
        else:
            assert color_mapping[score_type] == default_color


def test_bar_configs_incomplete_colors(make_mock_df, score_col, score_type_col, rank_col):
    """
    Test that the bar configuration fails if the colors dict does not cover the score types in the
    data
    """
    with pytest.raises(KeyError):
        _ = shift_bars(
            make_mock_df(),
            rank_col=rank_col,
            score_col=score_col,
            score_type_col=score_type_col,
            bars=InternalBarConfig(colors={"lower_score": "red"}, orders=None, symbols={}),
        )


# from_bar_config picks its config class from `attributed_by`: BarConfig (reference/comparison) for
# frequency-attributed bars, ScoreSignBarConfig (reinforce/offset) for score-sign-attributed bars.
@pytest.mark.parametrize(
    "config_cls, attributed_by, attribution_fields",
    [
        (BarConfig, "frequency", ("reference", "comparison")),
        (ScoreSignBarConfig, "score_sign", ("reinforce", "offset")),
    ],
)
def test_bar_config_default_colors(
    token_col, score_col, score_type_col, rank_col, config_cls, attributed_by, attribution_fields
):
    """
    `InternalBarConfig.from_bar_config` colors the bars with the config's defaults and, since bar
    plots don't stack, enforces no stacking order
    """
    first, second = attribution_fields
    df = pl.DataFrame(
        {
            token_col: ["a", "b"],
            rank_col: [1, 2],
            score_type_col: [first, second],
            score_col: [1.0, -1.0],
        }
    )

    chart = shift_bars(
        df,
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        bars=InternalBarConfig.from_bar_config(None, attributed_by=attributed_by),
    )

    chart_json = chart.to_dict()
    scale = chart_json["encoding"]["color"]["scale"]

    color_mapping = dict(zip(scale["domain"], scale["range"]))
    assert color_mapping == {field: getattr(config_cls(), field) for field in attribution_fields}
    assert "order" not in chart_json["encoding"]


@pytest.mark.parametrize("as_dict", [False, True], ids=["object", "dict"])
@pytest.mark.parametrize(
    "config_cls, attributed_by, attribution_fields",
    [
        (BarConfig, "frequency", ("reference", "comparison")),
        (ScoreSignBarConfig, "score_sign", ("reinforce", "offset")),
    ],
)
def test_bar_config_custom_colors(
    token_col,
    score_col,
    score_type_col,
    rank_col,
    config_cls,
    attributed_by,
    attribution_fields,
    as_dict,
):
    """
    A custom config overrides the first bar's color while keeping the second at default, whether
    passed as a config object or as a dict of the same fields
    """
    first, second = attribution_fields

    overrides = {first: "#000000"}
    config = overrides if as_dict else config_cls(**overrides)

    df = pl.DataFrame(
        {
            token_col: ["a", "b"],
            rank_col: [1, 2],
            score_type_col: [first, second],
            score_col: [1.0, -1.0],
        }
    )

    chart = shift_bars(
        df,
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        bars=InternalBarConfig.from_bar_config(config, attributed_by=attributed_by),
    )

    scale = chart.to_dict()["encoding"]["color"]["scale"]
    color_mapping = dict(zip(scale["domain"], scale["range"]))
    assert color_mapping == {first: "#000000", second: getattr(config_cls(), second)}


def test_total_shift_config_default_color(token_col, score_col, score_type_col, rank_col):
    """
    `InternalBarConfig.from_total_shift_config` colors the single grand-total bar with the
    TotalShiftConfig's default `total` color
    """
    df = pl.DataFrame(
        {
            token_col: ["a"],
            rank_col: [1],
            score_type_col: ["total"],
            score_col: [1.0],
        }
    )

    chart = shift_bars(
        df,
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        bars=InternalBarConfig.from_total_shift_config(None),
    )

    chart_json = chart.to_dict()
    scale = chart_json["encoding"]["color"]["scale"]

    color_mapping = dict(zip(scale["domain"], scale["range"]))
    assert color_mapping == {"total": TotalShiftConfig().total}
    assert "order" not in chart_json["encoding"]


@pytest.mark.parametrize("as_dict", [False, True])
def test_total_shift_config_custom_color(token_col, score_col, score_type_col, rank_col, as_dict):
    """
    A custom config overrides the grand-total bar's `total` color, whether passed as a
    TotalShiftConfig object or as a dict of the same fields
    """
    config = {"total": "#000000"} if as_dict else TotalShiftConfig(total="#000000")
    df = pl.DataFrame(
        {
            token_col: ["a"],
            rank_col: [1],
            score_type_col: ["total"],
            score_col: [1.0],
        }
    )

    chart = shift_bars(
        df,
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        bars=InternalBarConfig.from_total_shift_config(config),
    )

    scale = chart.to_dict()["encoding"]["color"]["scale"]
    color_mapping = dict(zip(scale["domain"], scale["range"]))
    assert color_mapping == {"total": "#000000"}


@pytest.mark.parametrize(
    "make_internal_bar_config",
    [
        lambda config: InternalBarConfig.from_bar_config(config, attributed_by="frequency"),
        lambda config: InternalBarConfig.from_shift_bar_config(config),
        lambda config: InternalBarConfig.from_total_shift_config(config),
    ],
    ids=["from_bar_config", "from_shift_bar_config", "from_total_shift_config"],
)
def test_from_config_rejects_invalid_dict_keys(make_internal_bar_config):
    """
    Passing a dict with field names the config doesn't have raises the `_resolve_config` ValueError
    """
    with pytest.raises(ValueError, match="Invalid .* key"):
        make_internal_bar_config({"not_a_field": "red", "also_bad": "blue"})


@pytest.mark.parametrize(
    "make_internal_bar_config, wrong_config",
    [
        (
            lambda config: InternalBarConfig.from_bar_config(config, attributed_by="frequency"),
            ShiftBarConfig(),
        ),
        (lambda config: InternalBarConfig.from_shift_bar_config(config), BarConfig()),
        (lambda config: InternalBarConfig.from_total_shift_config(config), BarConfig()),
    ],
    ids=["from_bar_config", "from_shift_bar_config", "from_total_shift_config"],
)
def test_from_config_rejects_wrong_config_type(make_internal_bar_config, wrong_config):
    """
    Passing a config of the wrong class raises the `_resolve_config` TypeError
    """
    with pytest.raises(TypeError, match="Expected a .* or a dict of its fields"):
        make_internal_bar_config(wrong_config)


def test_bar_outline_rendered_on_mark(token_col, score_col, score_type_col, rank_col):
    """
    The bar mark draws the outline color and width from the `InternalBarConfig`
    """
    df = pl.DataFrame({token_col: ["a"], rank_col: [1], score_type_col: ["x"], score_col: [1.0]})

    chart = shift_bars(
        df,
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        bars=InternalBarConfig(
            colors={"x": "red"}, orders=None, symbols={}, outline_color="blue", outline_width=2
        ),
    )

    mark = chart.to_dict()["mark"]
    assert (mark["stroke"], mark["strokeWidth"]) == ("blue", 2)


def test_bar_outline_absent_when_none(token_col, score_col, score_type_col, rank_col):
    """
    An outline color and width of `None` leave the mark's stroke to Altair's default
    """
    df = pl.DataFrame({token_col: ["a"], rank_col: [1], score_type_col: ["x"], score_col: [1.0]})

    chart = shift_bars(
        df,
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        bars=InternalBarConfig(
            colors={"x": "red"}, orders=None, symbols={}, outline_color=None, outline_width=None
        ),
    )

    mark = chart.to_dict()["mark"]
    assert "stroke" not in mark
    assert "strokeWidth" not in mark


@pytest.mark.parametrize(
    "make_internal_bar_config",
    [
        lambda config: InternalBarConfig.from_bar_config(config, attributed_by="frequency"),
        lambda config: InternalBarConfig.from_bar_config(config, attributed_by="score_sign"),
        lambda config: InternalBarConfig.from_shift_bar_config(config),
    ],
    ids=["BarConfig", "ScoreSignBarConfig", "ShiftBarConfig"],
)
def test_internal_bar_config_default_outline(make_internal_bar_config):
    """
    Without a user config, the bars take the default outline color and width
    """
    internal = make_internal_bar_config(None)
    assert (internal.outline_color, internal.outline_width) == (
        DEFAULT_BAR_OUTLINE_COLOR,
        DEFAULT_BAR_OUTLINE_WIDTH,
    )


@pytest.mark.parametrize("as_dict", [False, True], ids=["object", "dict"])
@pytest.mark.parametrize(
    "config_cls, make_internal_bar_config",
    [
        (BarConfig, lambda c: InternalBarConfig.from_bar_config(c, attributed_by="frequency")),
        (
            ScoreSignBarConfig,
            lambda c: InternalBarConfig.from_bar_config(c, attributed_by="score_sign"),
        ),
        (ShiftBarConfig, lambda c: InternalBarConfig.from_shift_bar_config(c)),
        (TotalShiftConfig, lambda c: InternalBarConfig.from_total_shift_config(c)),
    ],
    ids=["BarConfig", "ScoreSignBarConfig", "ShiftBarConfig", "TotalShiftConfig"],
)
def test_internal_bar_config_custom_outline(config_cls, make_internal_bar_config, as_dict):
    """
    A user config's outline color and width carry through to the `InternalBarConfig`, whether passed
    as a config object or as a dict of the same fields
    """
    overrides = {"outline_color": "blue", "outline_width": 2}
    config = overrides if as_dict else config_cls(**overrides)

    internal = make_internal_bar_config(config)
    assert (internal.outline_color, internal.outline_width) == ("blue", 2)


@pytest.mark.parametrize(
    "internal_bar_config",
    [
        InternalBarConfig.from_bar_config(None, attributed_by="frequency"),
        InternalBarConfig.from_bar_config(None, attributed_by="score_sign"),
        InternalBarConfig.from_shift_bar_config(None),
        InternalBarConfig.from_total_shift_config(None),
    ],
    ids=["BarConfig", "ScoreSignBarConfig", "ShiftBarConfig", "TotalShiftConfig"],
)
def test_internal_bar_config_colors_exclude_bar_outline_fields(internal_bar_config):
    """
    The bar outline fields style the bars and are not score types, so they have no bar color
    """
    assert BAR_OUTLINE_FIELDS.isdisjoint(internal_bar_config.colors)


def test_internal_bar_config_from_bar_config_has_no_orders():
    """
    `InternalBarConfig.from_bar_config()` produces an internal config without stacking orders, since
    bar plots don't stack
    """
    internal = InternalBarConfig.from_bar_config(None, attributed_by="frequency")
    assert internal.orders is None
    assert set(internal.colors.keys()) == {"reference", "comparison"}


def test_internal_bar_config_from_shift_bar_config_uses_internal_orders():
    """
    `InternalBarConfig.from_shift_bar_config()` produces an internal config whose `orders` equals
    `_SHIFT_TYPE_ORDERS` — i.e., stacking order is controlled internally, not by the user
    """
    internal = InternalBarConfig.from_shift_bar_config(None)
    assert internal.orders == _SHIFT_TYPE_ORDERS
    assert set(internal.colors.keys()) == set(SHIFT_SCORE_TYPES)
