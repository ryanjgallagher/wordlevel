import altair as alt
import polars as pl
import pytest

from polars.testing import assert_frame_equal

from wordlevel.configs import LabelConfig
from wordlevel.core.catalog.plot.shift.render.scales import (
    shift_scales,
    get_scale_limits,
    get_bar_lengths,
)


def _limits(df, *, score_col, rank_col, token_col, **kwargs) -> tuple[float, float]:
    """Gets the limits produced by get_scale_limits"""
    row = get_scale_limits(
        df, score_col=score_col, rank_col=rank_col, token_col=token_col, **kwargs
    ).row(0, named=True)

    return row["positive_limit"], row["negative_limit"]


def _chart_positive_limit(chart: alt.LayerChart) -> float:
    """Gets the positive_limit from a shift_scales chart, which applies max_rank filtering and config
    resolution that get_scale_limits (and so _limits) does not."""
    # chart datasets are of the form {arbitrary_key: [{rank: 1, positive_limit: N, negative_limit: -N}]}
    return list(chart.to_dict()["datasets"].values())[0][0]["positive_limit"]


def test_get_bar_lengths(score_col, rank_col, token_col):
    """ """
    df_input = pl.DataFrame(
        {
            token_col: ["w1", "w1", "w1", "w2", "w2", "w2", "w3", "w3", "w3", "w4"],
            rank_col: [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
            score_col: [5, -3, 4, -6, -2, 7, 1, 0, -1, 7],
        }
    )

    df_expected = pl.DataFrame(
        {token_col: ["w1", "w2", "w3", "w4"], rank_col: [1, 2, 3, 4], "bar_length": [9, 8, 1, 7]}
    )

    df_actual = get_bar_lengths(
        df_input, score_col=score_col, rank_col=rank_col, token_col=token_col
    )

    assert_frame_equal(
        df_actual.select(df_expected.columns),
        df_expected,
        check_row_order=False,
        check_column_order=False,
        check_dtypes=False,
    )


def test_on_bar_ends_reserves_label_space(score_col, rank_col, token_col):
    """A label at the bar tip (on_bar_ends=True) extends the domain, whereas one pinned at the zero
    line does not, so the same data yields a wider limit when labels sit on the bar ends."""
    df = pl.DataFrame({token_col: ["short", "muchlongerword"], rank_col: [1, 2], score_col: [5, 4]})
    kwargs = dict(
        score_col=score_col, rank_col=rank_col, token_col=token_col, size=600, fontsize=11
    )

    on_ends, _ = _limits(df, on_bar_ends=True, **kwargs)
    at_zero, _ = _limits(df, on_bar_ends=False, **kwargs)

    assert on_ends > at_zero


def test_domain_is_symmetric_when_signs_mixed(score_col, rank_col, token_col):
    """Mixed-sign contributions produce a symmetric [-L, L] domain."""
    df = pl.DataFrame({token_col: ["w1", "w2"], rank_col: [1, 2], score_col: [5, -4]})

    positive, negative = _limits(
        df, score_col=score_col, rank_col=rank_col, token_col=token_col, size=600, fontsize=11
    )

    assert negative == -positive


def test_domain_is_symmetric_when_bars_negative_but_net_positive(score_col, rank_col, token_col):
    """A token can have all net positive contributions while still having a negative bar component
    (e.g. counteracted components in a JSD shift). Those negative bars still need room, so the
    domain must be symmetric even though every token's collapsed net is positive."""
    df = pl.DataFrame(
        {
            token_col: ["w1", "w1", "w2", "w2"],
            rank_col: [1, 1, 2, 2],
            score_col: [10, -3, 8, -2],
        }
    )

    positive, negative = _limits(
        df, score_col=score_col, rank_col=rank_col, token_col=token_col, size=600, fontsize=11
    )

    assert negative == -positive


def test_domain_is_one_sided_when_all_positive(score_col, rank_col, token_col):
    """All-positive contributions produce a one-sided [0, L] domain."""
    df = pl.DataFrame({token_col: ["w1", "w2"], rank_col: [1, 2], score_col: [5, 4]})

    _, negative = _limits(
        df, score_col=score_col, rank_col=rank_col, token_col=token_col, size=600, fontsize=11
    )

    assert negative == 0


def test_scale_factor_scales_limit_linearly(score_col, rank_col, token_col):
    """scale_factor multiplies the computed limit."""
    scale_factor = 3

    df = pl.DataFrame({token_col: ["w1", "w2"], rank_col: [1, 2], score_col: [5, 4]})
    kwargs = dict(
        score_col=score_col, rank_col=rank_col, token_col=token_col, size=600, fontsize=11
    )

    base, _ = _limits(df, scale_factor=1, **kwargs)
    scaled, _ = _limits(df, scale_factor=scale_factor, **kwargs)

    assert scaled == pytest.approx(scale_factor * base)


def test_wider_label_widens_limit(score_col, rank_col, token_col):
    """A longer token reserves more label space at the bar tip, widening the limit."""
    kwargs = dict(
        score_col=score_col, rank_col=rank_col, token_col=token_col, size=600, fontsize=11
    )

    narrow, _ = _limits(pl.DataFrame({token_col: ["a"], rank_col: [1], score_col: [5]}), **kwargs)
    wide, _ = _limits(
        pl.DataFrame({token_col: ["a" * 20], rank_col: [1], score_col: [5]}), **kwargs
    )

    assert wide > narrow


def test_returns_chart(make_mock_df, score_col, rank_col, token_col):
    """
    Sanity check that the function returns an Altair chart
    """
    chart = shift_scales(
        make_mock_df(), score_col=score_col, rank_col=rank_col, token_col=token_col, width=10
    )

    assert isinstance(chart, alt.LayerChart)


@pytest.mark.parametrize("vertical, wrong_size_col", [(True, "height"), (False, "width")])
def test_error_if_size_not_specified(
    make_mock_df, score_col, rank_col, token_col, vertical, wrong_size_col
):
    """ """
    with pytest.raises(ValueError):
        _ = shift_scales(
            make_mock_df(),
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            vertical=vertical,
            # Dynamic key: the type checker can't tell wrong_size_col is width/height
            **{wrong_size_col: 10},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("vertical", [True, False])
def test_chart_orientation(make_mock_df, score_col, rank_col, token_col, vertical):
    """
    Test that the vertical parameter correctly swaps the X and Y axes
    for the scale layers.
    """
    chart = shift_scales(
        make_mock_df(),
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
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


def test_max_rank_filter(score_col, rank_col, token_col):
    """
    Test that max_rank filters the data before the domain is computed. w1 has the largest
    contribution but a rank beyond max_rank, so filtering it out must shrink the limit.
    """
    df_mock = pl.DataFrame({token_col: ["w1", "w2"], rank_col: [100, 1], score_col: [100, 5]})

    def positive_limit(filter_max_rank: bool) -> float:
        chart = shift_scales(
            df_mock,
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            max_rank=2,
            filter_max_rank=filter_max_rank,
            width=600,
            config={"size": 11},
        )
        return _chart_positive_limit(chart)

    assert positive_limit(filter_max_rank=True) < positive_limit(filter_max_rank=False)


def test_default_fontsize(make_mock_df, score_col, rank_col, token_col):
    """ """
    chart_config = shift_scales(
        make_mock_df(), score_col, rank_col, token_col, width=10, config=LabelConfig()
    )
    chart_none = shift_scales(
        make_mock_df(), score_col, rank_col, token_col, width=10, config=LabelConfig(size=None)
    )

    data_config = list(chart_config.to_dict()["datasets"].values())[0]
    data_none = list(chart_none.to_dict()["datasets"].values())[0]
    limit_config = data_config[0]["positive_limit"]
    limit_none = data_none[0]["positive_limit"]

    # Limits should be the same if the chart defaulted correctly
    assert limit_config == limit_none


def test_layers_are_invisible(make_mock_df, score_col, rank_col, token_col):
    """
    Tests that the scale layers are invisible (opacity 0) and have tooltips disabled.
    """
    chart = shift_scales(make_mock_df(), score_col, rank_col, token_col, width=10)

    layers = chart.to_dict()["layer"]

    for layer in layers:
        mark = layer["mark"]
        assert isinstance(mark, dict)
        assert mark["opacity"] == 0

        encoding = layer["encoding"]
        assert "tooltip" in encoding
        assert encoding["tooltip"]["value"] is None
