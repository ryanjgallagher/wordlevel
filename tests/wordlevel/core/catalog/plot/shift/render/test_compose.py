from typing import cast

import altair as alt
import polars as pl
import pytest

from wordlevel.configs import BorderConfig, TotalShiftConfig, ZeroLineConfig
from wordlevel.core.catalog.plot.shift.render.shift import shift
from wordlevel.core.catalog.plot.shift.render.totals import _LABEL_COL, _SCORE_TYPE_COL


def _shift(
    df, score_col, rank_col, token_col, score_type_col, **kwargs
) -> alt.LayerChart | alt.VConcatChart | alt.HConcatChart:
    """Call `shift` with the standard column mapping and test-sized dimensions."""
    kwargs.setdefault("max_rank", 10)
    kwargs.setdefault("width", 100)
    kwargs.setdefault("height", 100)
    return shift(
        df,
        score_col=score_col,
        rank_col=rank_col,
        token_col=token_col,
        score_type_col=score_type_col,
        **kwargs,
    )


def _panels(chart: alt.TopLevelMixin) -> list[dict]:
    """The per-panel Vega-Lite specs of a `show_totals` concat chart."""
    spec = chart.to_dict()
    return cast(list[dict], spec.get("vconcat") or spec.get("hconcat"))


def _border_edges(panel: dict) -> set[tuple]:
    """The (domainColor, domainWidth) of every drawn border edge in a panel spec.

    The border edges are the non-axis frame layers `shift_borders` appends after the base group.
    """
    edges = set()
    for layer in panel["layer"][1:]:
        encoding = layer["encoding"]
        axis = (encoding.get("x") or encoding.get("y"))["axis"]
        edges.add((axis["domainColor"], axis["domainWidth"]))
    
    return edges


def _zero_line_mark(panel: dict) -> dict:
    """The zero-line rule mark (the 4th base component) of a panel spec."""
    return panel["layer"][0]["layer"][3]["mark"]


def _panel_bars_data(panel: alt.TopLevelMixin) -> pl.DataFrame:
    """The dataframe plotted by a panel's bars layer."""
    return cast(pl.DataFrame, cast(alt.LayerChart, panel).layer[0].layer[0].data)


class TestShowTotals:
    @pytest.mark.parametrize(
        "vertical, concat_attr, chart_type",
        [
            (True, "vconcat", alt.VConcatChart),
            (False, "hconcat", alt.HConcatChart),
        ],
    )
    def test_returns_three_panel_concat(
        self,
        make_mock_df,
        score_col,
        rank_col,
        token_col,
        score_type_col,
        vertical,
        concat_attr,
        chart_type,
    ):
        """`show_totals` stacks three panels (per-type totals, grand total, main chart) with a
        concat matching the orientation."""
        chart = _shift(
            make_mock_df(n=50),
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            score_type_col=score_type_col,
            show_totals=True,
            vertical=vertical,
        )

        assert isinstance(chart, chart_type)
        assert len(getattr(chart, concat_attr)) == 3

    def test_no_totals_returns_single_layer_chart(
        self, make_mock_df, score_col, rank_col, token_col, score_type_col
    ):
        """Without `show_totals`, `shift` returns a single layer chart rather than a concat."""
        chart = _shift(
            make_mock_df(n=50),
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            score_type_col=score_type_col,
            show_totals=False,
        )

        assert isinstance(chart, alt.LayerChart)

    def test_color_scales_are_independent(
        self, make_mock_df, score_col, rank_col, token_col, score_type_col
    ):
        """The stacked panels resolve their color scales independently so their legends do not
        merge into one shared scale."""
        chart = _shift(
            make_mock_df(n=50),
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            score_type_col=score_type_col,
            show_totals=True,
        )

        assert chart.to_dict()["resolve"]["scale"]["color"] == "independent"

    def test_panel_order_is_types_then_grand_then_main(
        self, make_mock_df, score_col, rank_col, token_col, score_type_col
    ):
        """The panels are ordered [per-type totals, grand total, main]: the last panel is the full
        word-shift chart, the middle panel is the single grand-total bar."""
        chart = _shift(
            make_mock_df(n=50),
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            score_type_col=score_type_col,
            show_totals=True,
            max_rank=10,
        )

        type_totals, grand_total, main = cast(alt.VConcatChart, chart).vconcat

        # First panel: one bar per score type (several types, no token column)
        type_bars = _panel_bars_data(type_totals)
        assert token_col not in type_bars.columns
        assert type_bars[_SCORE_TYPE_COL].n_unique() > 1

        # Middle panel: the single grand-total bar
        assert _panel_bars_data(grand_total).height == 1

        # Last panel: the full word-shift chart up to max_rank
        main_bars = _panel_bars_data(main)
        assert token_col in main_bars.columns
        assert main_bars[rank_col].max() == 10

    def test_forwards_total_shift_config(
        self, make_mock_df, score_col, rank_col, token_col, score_type_col
    ):
        """`shift` forwards `total_shift_config` into the grand-total panel."""
        chart = _shift(
            make_mock_df(n=50),
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            score_type_col=score_type_col,
            show_totals=True,
            total_shift_config=TotalShiftConfig(total_symbol="Ω"),
        )

        grand_bars = _panel_bars_data(cast(alt.VConcatChart, chart).vconcat[1])
        assert grand_bars[_LABEL_COL][0] == "Ω"


class TestFrameConsistency:
    def test_borders_same_color_and_width_across_panels(
        self, make_mock_df, score_col, rank_col, token_col, score_type_col
    ):
        """Every border edge on every panel shares one color and width, so the stacked frame reads
        as a single continuous border rather than mismatched per-panel edges."""
        chart = _shift(
            make_mock_df(n=50),
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            score_type_col=score_type_col,
            show_totals=True,
            border_config=BorderConfig(color="red", width=2),
        )

        edges = set().union(*(_border_edges(panel) for panel in _panels(chart)))
        assert edges == {("red", 2)}

    def test_zero_line_same_color_and_width_across_panels(
        self, make_mock_df, score_col, rank_col, token_col, score_type_col
    ):
        """The zero line is drawn identically (color and width) on the main chart and both totals
        panels."""
        chart = _shift(
            make_mock_df(n=50),
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            score_type_col=score_type_col,
            show_totals=True,
            zero_line_config=ZeroLineConfig(color="blue", width=3),
        )

        marks = [_zero_line_mark(panel) for panel in _panels(chart)]
        assert {(mark["color"], mark["strokeWidth"]) for mark in marks} == {("blue", 3)}
