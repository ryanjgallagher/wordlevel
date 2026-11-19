import altair as alt
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from wordlevel.configs import (
    DEFAULT_ATTRIBUTION_COLORS,
    DEFAULT_BAR_OUTLINE_COLOR,
    DEFAULT_BAR_OUTLINE_WIDTH,
    DEFAULT_GRAND_TOTAL_COLOR,
    DEFAULT_SCORE_AXIS_TITLE,
    PERCENT_SCORE_AXIS_TITLE,
    BarColorConfig,
    BarConfig,
    RankAxisConfig,
    ScoreAxisConfig,
    ShiftColorConfig,
)
from wordlevel.constants import (
    COUNTERACTED_SUFFIX,
    GRAND_TOTAL_SCORE_TYPE,
    SHIFT_BASE_SCORE_TYPES,
    SHIFT_SCORE_TYPES,
)
from wordlevel.plots.core.shift.bars import (
    _SHIFT_TYPE_ORDERS,
    InternalBarConfig,
    InternalScoreAxisConfig,
    shift_bars,
)
from wordlevel.plots.core.shift.border import InternalBorderConfig


def test_returns_chart(make_mock_df, make_bars_configs, score_col, score_type_col, rank_col):
    """
    Sanity check that the function returns an Altair chart
    """
    chart = shift_bars(
        make_mock_df(),
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        **make_bars_configs(),
    )

    assert isinstance(chart, alt.Chart)


def test_chart_data_and_encodings(
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col
):
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
        **make_bars_configs(),
    )

    # The shift bar config's stacking orders add a `stacked_bar_order` column to the input data
    df_chart = pl.DataFrame(chart.data)
    assert_frame_equal(df, df_chart.select(df.columns))

    chart_json = chart.to_dict()
    encoding = chart_json["encoding"]

    assert encoding["x"]["field"] == score_col
    assert encoding["x"]["type"] == "quantitative"

    assert encoding["y"]["field"] == rank_col
    assert encoding["y"]["type"] == "ordinal"

    assert encoding["color"]["field"] == score_type_col
    assert encoding["color"]["type"] == "nominal"


@pytest.mark.parametrize("vertical", [True, False])
def test_chart_orientation(
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col, vertical
):
    """
    Test that the vertical parameter correctly swaps the X and Y axes.
    """
    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=vertical,
        **make_bars_configs(),
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
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col, vertical, bar_width
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
        **make_bars_configs(rank_axis_config=RankAxisConfig(bar_width=bar_width)),
    )

    spec = chart.to_dict()
    rank_dim = "height" if vertical else "width"
    assert spec[rank_dim] == {"step": bar_width}


@pytest.mark.parametrize("vertical", [True, False])
def test_rank_dimension_ignores_explicit_size(
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col, vertical
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
        **make_bars_configs(),
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
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col, vertical
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
        **make_bars_configs(),
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
        (InternalBorderConfig(), "#888"),
        # Explicitly opting out (color=None) defers to Altair's per-element defaults (keys absent)
        (InternalBorderConfig(color=None), None),
        (InternalBorderConfig(color="black", width=2), "black"),
    ],
)
def test_border_config(
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col, border, expected_color
):
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
        **make_bars_configs(border_config=border),
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
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col, axis_config_cls, channel
):
    """
    The native axis domain width is owned by the axis config's own `width`; it does not fall back to
    `BorderConfig.width`, which governs only the opposite border edges.
    """
    config = axis_config_cls(width=3)
    kwargs = {"rank_axis_config": config} if channel == "y" else {"score_axis_config": config}

    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=True,
        **make_bars_configs(
            # The border width differs from the axis width, so it must not leak in
            border_config=InternalBorderConfig(width=1),
            **kwargs,
        ),
    )

    axis = chart.to_dict()["encoding"][channel]["axis"]
    assert axis["domainWidth"] == 3


@pytest.mark.parametrize("filter_max_rank", [True, False])
def test_max_rank_filter(
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col, filter_max_rank
):
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
        **make_bars_configs(),
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
    make_bars_configs,
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
    # The parametrized `Config` selects the rank or score axis
    is_rank = issubclass(Config, RankAxisConfig)
    axis = "y" if is_rank else "x"

    config = Config(
        title=title, title_fontsize=title_fontsize, label_fontsize=label_fontsize, grid=grid
    )
    axis_kwargs = {"rank_axis_config": config} if is_rank else {"score_axis_config": config}
    chart = shift_bars(
        make_mock_df(),
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        vertical=True,
        **make_bars_configs(**axis_kwargs),
    )

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


def test_axis_default_configs(make_mock_df, make_bars_configs, score_col, score_type_col, rank_col):
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
        **make_bars_configs(),
    )
    chart_json = chart.to_dict()

    rank_axis_config = chart_json["encoding"]["y"]["axis"]
    assert rank_axis_config["title"] == rank_config.title
    assert rank_axis_config["grid"] == rank_config.grid

    score_axis_config = chart_json["encoding"]["x"]["axis"]
    assert score_axis_config["title"] == score_config.title
    assert score_axis_config["grid"] == score_config.grid


@pytest.mark.parametrize(
    "axis_config_cls, channel", [(RankAxisConfig, "y"), (InternalScoreAxisConfig, "x")]
)
def test_axis_text_and_grid_colors(
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col, axis_config_cls, channel
):
    """
    label_color, title_color, and grid_color on the rank/score axis configs pass through to the
    corresponding alt.Axis keys. These are the per-axis text/gridline colors, distinct from the
    structural stroke color set by BorderConfig (domainColor/tickColor).
    """
    config = axis_config_cls(label_color="#111111", title_color="#222222", grid_color="#333333")

    kwargs = {"rank_axis_config": config} if channel == "y" else {"score_axis_config": config}
    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=True,
        **make_bars_configs(**kwargs),
    )

    axis = chart.to_dict()["encoding"][channel]["axis"]
    assert axis["labelColor"] == "#111111"
    assert axis["titleColor"] == "#222222"
    assert axis["gridColor"] == "#333333"


@pytest.mark.parametrize(
    "axis_config_cls, channel", [(RankAxisConfig, "y"), (InternalScoreAxisConfig, "x")]
)
def test_axis_colors_absent_by_default(
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col, axis_config_cls, channel
):
    """
    When the per-axis colors are left at their default (None), the corresponding keys are not
    emitted, so Altair falls back to its own defaults.
    """
    kwargs = (
        {"rank_axis_config": axis_config_cls()}
        if channel == "y"
        else {"score_axis_config": axis_config_cls()}
    )
    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=True,
        **make_bars_configs(**kwargs),
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
    make_bars_configs,
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
    kwargs = {"rank_axis_config": config} if channel == "y" else {"score_axis_config": config}

    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=True,
        **make_bars_configs(border_config=InternalBorderConfig(color=border_color), **kwargs),
    )

    axis = chart.to_dict()["encoding"][channel]["axis"]

    # Both the rank and score axes draw their domain, so domain and tick colors behave identically
    for key, expected in (("domainColor", expected_domain), ("tickColor", expected_tick)):
        if expected is None:
            assert key not in axis
        else:
            assert axis[key] == expected


def test_axis_colors_are_per_axis(
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col
):
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
        **make_bars_configs(
            rank_axis_config=RankAxisConfig(axis_color="red", tick_color="green"),
            score_axis_config=InternalScoreAxisConfig(axis_color="blue", tick_color="orange"),
            border_config=InternalBorderConfig(color="black"),
        ),
    )

    encoding = chart.to_dict()["encoding"]
    rank_axis_config = encoding["y"]["axis"]  # rank axis is y when vertical
    score_axis_config = encoding["x"]["axis"]

    # Both axes draw their own domain and ticks, independently colored
    assert rank_axis_config["domainColor"] == "red"
    assert rank_axis_config["tickColor"] == "green"
    assert score_axis_config["domainColor"] == "blue"
    assert score_axis_config["tickColor"] == "orange"


@pytest.mark.parametrize("all_positive_labels", [True, False])
def test_score_axis_all_positive_labels(
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col, all_positive_labels
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
        **make_bars_configs(
            score_axis_config=InternalScoreAxisConfig(all_positive_labels=all_positive_labels)
        ),
    )

    chart_json = chart.to_dict()
    score_axis_config = chart_json["encoding"]["x"]["axis"]

    if all_positive_labels:
        assert score_axis_config["labelExpr"] == "abs(datum.value)"
    else:
        assert "labelExpr" not in score_axis_config


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
    make_bars_configs,
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
        **make_bars_configs(
            score_axis_config=InternalScoreAxisConfig(
                all_positive_labels=all_positive_labels, percent=percent
            )
        ),
    )

    score_axis_config = chart.to_dict()["encoding"]["x"]["axis"]

    if expected_label_expr is None:
        assert "labelExpr" not in score_axis_config
    else:
        assert score_axis_config["labelExpr"] == expected_label_expr


@pytest.mark.parametrize(
    "config, percent, expected_title",
    [
        # The default title becomes the percent title only when shown as percentages
        (ScoreAxisConfig(), False, DEFAULT_SCORE_AXIS_TITLE),
        (ScoreAxisConfig(), True, PERCENT_SCORE_AXIS_TITLE),
        # A custom title is preserved, even when shown as percentages
        (ScoreAxisConfig(title="Custom"), True, "Custom"),
        # An explicitly hidden title (None) stays hidden (no title rendered)
        (ScoreAxisConfig(title=None), True, None),
    ],
)
def test_score_axis_title_resolved_and_rendered(
    make_mock_df,
    make_bars_configs,
    score_col,
    score_type_col,
    rank_col,
    config,
    percent,
    expected_title,
):
    """The resolved title (percent only swaps the default) is carried through to the chart."""
    chart = shift_bars(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        score_type_col=score_type_col,
        vertical=True,
        **make_bars_configs(
            score_axis_config=InternalScoreAxisConfig.from_config(
                config, all_positive_labels=False, percent=percent
            )
        ),
    )

    score_axis_config = chart.to_dict()["encoding"]["x"]["axis"]
    # Indexing (not `.get`) so a missing title, which Vega-Lite renders as the field name, fails
    assert score_axis_config["title"] == expected_title


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
    make_bars_configs,
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
        **make_bars_configs(
            rank_axis_config=RankAxisConfig(
                label_tick_values=tick_values,
                label_tick_step=tick_step,
                label_first_tick=first_tick,
            )
        ),
    )

    chart_json = chart.to_dict()
    axis = chart_json["encoding"]["y"]["axis"]

    if expected_output is None:
        assert "values" not in axis
    else:
        assert axis["values"] == expected_output


def _expected_shift_colors(config: ShiftColorConfig) -> dict[str, str]:
    """
    The color expected for every score type. The config declares only the six base colors, so each
    `*_counteracted` score type is expected to take its base score type's color
    """
    expected = {}
    for score_type in SHIFT_BASE_SCORE_TYPES:
        color = getattr(config, score_type)
        expected[score_type] = color
        expected[f"{score_type}{COUNTERACTED_SUFFIX}"] = color

    return expected


def test_shift_bar_configs_default(
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col
):
    """
    Test that the default shift bar configuration applies the ShiftColorConfig defaults
    """
    chart = shift_bars(
        make_mock_df(),
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        **make_bars_configs(
            bar_config=InternalBarConfig.from_shift_colors(
                config=ShiftColorConfig(), bar_config=BarConfig()
            )
        ),
    )

    chart_json = chart.to_dict()
    scale = chart_json["encoding"]["color"]["scale"]

    default_colors = _expected_shift_colors(ShiftColorConfig())

    assert set(scale["domain"]) == set(default_colors.keys())
    assert set(scale["range"]) == set(default_colors.values())

    color_mapping = dict(zip(scale["domain"], scale["range"]))
    for score_type, expected_color in default_colors.items():
        assert color_mapping[score_type] == expected_color

    assert chart_json["encoding"]["order"]["field"] == "stacked_bar_order"


def test_shift_bar_configs_custom_colors(
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col
):
    """
    Test that a custom ShiftColorConfig overrides individual colors while keeping the rest at
    default. Overriding a base color also recolors that score type's `*_counteracted` bar
    """
    config = ShiftColorConfig(lower_score="#000000", more_of_positive="#999999")

    chart = shift_bars(
        make_mock_df(),
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        **make_bars_configs(
            bar_config=InternalBarConfig.from_shift_colors(config=config, bar_config=BarConfig())
        ),
    )

    chart_json = chart.to_dict()
    scale = chart_json["encoding"]["color"]["scale"]

    expected_colors = _expected_shift_colors(config)

    assert dict(zip(scale["domain"], scale["range"])) == expected_colors


def test_shift_bar_configs_fade_counteracted_bars(
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col
):
    """
    Test that the `*_counteracted` bars of a shift plot are faded to `counteracted_opacity` while
    every base bar stays opaque, and that the fade carries no legend of its own
    """
    config = ShiftColorConfig(counteracted_opacity=0.2)

    chart = shift_bars(
        make_mock_df(),
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        **make_bars_configs(
            bar_config=InternalBarConfig.from_shift_colors(config=config, bar_config=BarConfig())
        ),
    )

    fill_opacity = chart.to_dict()["encoding"]["fillOpacity"]
    opacity_mapping = dict(zip(fill_opacity["scale"]["domain"], fill_opacity["scale"]["range"]))

    expected_opacities = {
        score_type: 0.2 if score_type.endswith(COUNTERACTED_SUFFIX) else 1.0
        for score_type in SHIFT_SCORE_TYPES
    }

    assert opacity_mapping == expected_opacities
    assert fill_opacity["legend"] is None

    # Without an ordinal scale, Vega-Lite treats opacity as continuous and pairs each score type
    # with an evenly spaced opacity rather than the one above
    assert fill_opacity["scale"]["type"] == "ordinal"


def test_bar_configs_incomplete_colors(
    make_mock_df, make_bars_configs, score_col, score_type_col, rank_col
):
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
            **make_bars_configs(
                bar_config=InternalBarConfig(colors={"lower_score": "red"}, orders=None)
            ),
        )


# A `BarColorConfig` holds a color per attribution, but a comparison produces only one pair of
# them: reference/comparison when attributed by frequency or rank, reinforce/offset by score sign.
ATTRIBUTION_LABEL_PAIRS = [("reference", "comparison"), ("reinforce", "offset")]


@pytest.mark.parametrize(
    "attribution_labels", ATTRIBUTION_LABEL_PAIRS, ids=["frequency", "score_sign"]
)
def test_bar_colors_default(
    make_bars_configs, token_col, score_col, score_type_col, rank_col, attribution_labels
):
    """
    `InternalBarConfig.from_bar_colors` colors the bars with each attribution's default and, since
    bar plots don't stack and have no counteracted bars, enforces no stacking order and no fade
    """
    first, second = attribution_labels
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
        **make_bars_configs(
            bar_config=InternalBarConfig.from_bar_colors(
                config=BarColorConfig(),
                bar_config=BarConfig(),
                attribution_labels=attribution_labels,
            )
        ),
    )

    chart_json = chart.to_dict()
    scale = chart_json["encoding"]["color"]["scale"]

    color_mapping = dict(zip(scale["domain"], scale["range"]))
    assert color_mapping == {
        label: DEFAULT_ATTRIBUTION_COLORS[label] for label in attribution_labels
    }
    assert "order" not in chart_json["encoding"]
    assert "fillOpacity" not in chart_json["encoding"]


@pytest.mark.parametrize(
    "attribution_labels", ATTRIBUTION_LABEL_PAIRS, ids=["frequency", "score_sign"]
)
def test_bar_colors_custom(
    make_bars_configs, token_col, score_col, score_type_col, rank_col, attribution_labels
):
    """
    A custom config overrides the first bar's color while keeping the second at default
    """
    first, second = attribution_labels

    config = BarColorConfig(**{first: "#000000"})

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
        **make_bars_configs(
            bar_config=InternalBarConfig.from_bar_colors(
                config=config, bar_config=BarConfig(), attribution_labels=attribution_labels
            )
        ),
    )

    scale = chart.to_dict()["encoding"]["color"]["scale"]
    color_mapping = dict(zip(scale["domain"], scale["range"]))
    assert color_mapping == {first: "#000000", second: DEFAULT_ATTRIBUTION_COLORS[second]}


def test_color_configs_carry_grand_total_color():
    """
    Every color config carries the grand total bar's color, so that a totals panel can be drawn
    from the same config as the bars it totals
    """
    attribution_labels = ("reference", "comparison")

    defaults = (
        InternalBarConfig.from_bar_colors(
            config=BarColorConfig(), bar_config=BarConfig(), attribution_labels=attribution_labels
        ),
        InternalBarConfig.from_shift_colors(config=ShiftColorConfig(), bar_config=BarConfig()),
    )
    customs = (
        InternalBarConfig.from_bar_colors(
            config=BarColorConfig(grand_total="#000000"),
            bar_config=BarConfig(),
            attribution_labels=attribution_labels,
        ),
        InternalBarConfig.from_shift_colors(
            config=ShiftColorConfig(grand_total="#000000"), bar_config=BarConfig()
        ),
    )

    for config in defaults:
        assert config.colors[GRAND_TOTAL_SCORE_TYPE] == DEFAULT_GRAND_TOTAL_COLOR
    for config in customs:
        assert config.colors[GRAND_TOTAL_SCORE_TYPE] == "#000000"


def test_bar_outline_rendered_on_mark(
    make_bars_configs, token_col, score_col, score_type_col, rank_col
):
    """
    The bar mark draws the outline color and width from the `InternalBarConfig`
    """
    df = pl.DataFrame({token_col: ["a"], rank_col: [1], score_type_col: ["x"], score_col: [1.0]})

    chart = shift_bars(
        df,
        rank_col=rank_col,
        score_col=score_col,
        score_type_col=score_type_col,
        **make_bars_configs(
            bar_config=InternalBarConfig(
                colors={"x": "red"}, orders=None, outline_color="blue", outline_width=2
            )
        ),
    )

    mark = chart.to_dict()["mark"]
    assert (mark["stroke"], mark["strokeWidth"]) == ("blue", 2)


@pytest.mark.parametrize(
    "make_internal_bar_config",
    [
        lambda bar_config: InternalBarConfig.from_bar_colors(
            config=BarColorConfig(),
            bar_config=bar_config,
            attribution_labels=("reference", "comparison"),
        ),
        lambda bar_config: InternalBarConfig.from_shift_colors(
            config=ShiftColorConfig(), bar_config=bar_config
        ),
    ],
    ids=["from_bar_colors", "from_shift_colors"],
)
def test_internal_bar_config_default_outline(make_internal_bar_config):
    """
    With a default `BarConfig`, the bars take the default outline color and width
    """
    internal = make_internal_bar_config(BarConfig())
    assert (internal.outline_color, internal.outline_width) == (
        DEFAULT_BAR_OUTLINE_COLOR,
        DEFAULT_BAR_OUTLINE_WIDTH,
    )


@pytest.mark.parametrize(
    "make_internal_bar_config",
    [
        lambda bar_config: InternalBarConfig.from_bar_colors(
            config=BarColorConfig(),
            bar_config=bar_config,
            attribution_labels=("reference", "comparison"),
        ),
        lambda bar_config: InternalBarConfig.from_shift_colors(
            config=ShiftColorConfig(), bar_config=bar_config
        ),
    ],
    ids=["from_bar_colors", "from_shift_colors"],
)
def test_internal_bar_config_custom_outline(make_internal_bar_config):
    """
    The outline comes from the `BarConfig` rather than the color config
    """
    internal = make_internal_bar_config(BarConfig(outline_color="blue", outline_width=2))
    assert (internal.outline_color, internal.outline_width) == ("blue", 2)


def test_internal_bar_config_from_bar_colors_has_no_orders():
    """
    `InternalBarConfig.from_bar_colors()` produces an internal config without stacking orders, since
    bar plots don't stack
    """
    internal = InternalBarConfig.from_bar_colors(
        config=BarColorConfig(),
        bar_config=BarConfig(),
        attribution_labels=("reference", "comparison"),
    )
    assert internal.orders is None
    assert set(internal.colors) == {GRAND_TOTAL_SCORE_TYPE, "reference", "comparison"}


def test_internal_bar_config_from_shift_colors_uses_internal_orders():
    """
    `InternalBarConfig.from_shift_colors()` produces an internal config whose `orders` equals
    `_SHIFT_TYPE_ORDERS` — i.e., stacking order is controlled internally, not by the user
    """
    internal = InternalBarConfig.from_shift_colors(
        config=ShiftColorConfig(), bar_config=BarConfig()
    )
    assert internal.orders == _SHIFT_TYPE_ORDERS
    assert set(internal.colors) == {GRAND_TOTAL_SCORE_TYPE, *SHIFT_SCORE_TYPES}
