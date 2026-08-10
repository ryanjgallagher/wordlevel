import itertools
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


def _score_side_lines(panel: dict, score_col: str, vertical: bool) -> dict[str, list[float]]:
    """The widths of the frame lines on each side of a panel along the score-axis direction, keyed
    by side ('left'/'right' when horizontal, 'top'/'bottom' when vertical).

    A side gets a line from the panel's native score-axis domain and/or an opposite-score border
    edge; collecting both catches a missing seam (no line) and a doubled edge (two lines).
    """
    score_channel = "y" if not vertical else "x"
    default_orient = "left" if not vertical else "bottom"
    lines: dict[str, list[float]] = {}

    # Native score-axis domain (drawn on `orient`, which may be flipped to the opposite side)
    score_axis = panel["layer"][0]["layer"][0]["encoding"][score_channel].get("axis", {})
    if score_axis.get("domain"):
        side = score_axis.get("orient", default_orient)
        lines.setdefault(side, []).append(score_axis["domainWidth"])

    # Opposite-score border edges (the frame layers encoding the score field)
    for layer in panel["layer"][1:]:
        pos = layer["encoding"].get("x") or layer["encoding"].get("y")
        if pos["field"].startswith(score_col):
            lines.setdefault(pos["axis"]["orient"], []).append(pos["axis"]["domainWidth"])

    return lines


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

    @pytest.mark.parametrize("vertical, expected_orient", [(False, "right"), (True, None)])
    def test_totals_move_score_axis_opposite_only_when_horizontal(
        self,
        make_mock_df,
        score_col,
        rank_col,
        token_col,
        score_type_col,
        vertical,
        expected_orient,
    ):
        """With totals, the main chart's score axis is moved to the far (right) side only when
        horizontal, so it clears the totals panels placed to its left."""
        chart = _shift(
            make_mock_df(n=50),
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            score_type_col=score_type_col,
            show_totals=True,
            vertical=vertical,
        )

        main_panel = _panels(chart)[-1]  # the main chart is the last panel
        score_channel = "x" if vertical else "y"
        axis = main_panel["layer"][0]["layer"][0]["encoding"][score_channel].get("axis", {})
        assert axis.get("orient") == expected_orient


class TestFrameConsistency:
    @pytest.mark.parametrize("vertical", [True, False])
    def test_borders_same_color_and_width_across_panels(
        self, make_mock_df, score_col, rank_col, token_col, score_type_col, vertical
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
            vertical=vertical,
            border_config=BorderConfig(color="red", width=2),
        )

        edges = set().union(*(_border_edges(panel) for panel in _panels(chart)))
        assert edges == {("red", 2)}

    @pytest.mark.parametrize("vertical", [True, False])
    def test_zero_line_same_color_and_width_across_panels(
        self, make_mock_df, score_col, rank_col, token_col, score_type_col, vertical
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
            vertical=vertical,
            zero_line_config=ZeroLineConfig(color="blue", width=3),
        )

        marks = [_zero_line_mark(panel) for panel in _panels(chart)]
        assert {(mark["color"], mark["strokeWidth"]) for mark in marks} == {("blue", 3)}

    @pytest.mark.parametrize("vertical", [True, False])
    # The border box is off at width 0, so its outer edges are not drawn. The seams are structural
    # and are drawn either way
    @pytest.mark.parametrize("border_width, expected_outer", [(2, 1), (0, 0)])
    def test_every_score_direction_boundary_drawn_once(
        self,
        make_mock_df,
        score_col,
        rank_col,
        token_col,
        score_type_col,
        vertical,
        border_width,
        expected_outer,
    ):
        """Along the score-axis direction (where the panels are stacked), each inter-panel seam is
        drawn exactly once — catching both a missing divider and a doubled edge — in either
        orientation, whether or not the border box encloses the stack."""
        chart = _shift(
            make_mock_df(n=50),
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            score_type_col=score_type_col,
            show_totals=True,
            vertical=vertical,
            border_config=BorderConfig(color="red", width=border_width),
        )

        panels = _panels(chart)
        lines = [_score_side_lines(panel, score_col, vertical) for panel in panels]
        # Panels stack near -> far along the score axis (top->bottom vertical, left->right horizontal)
        near, far = ("top", "bottom") if vertical else ("left", "right")

        # The two outer edges of the stack. The far edge is the main chart's score axis, so it is
        # always drawn; the near edge belongs to the border box
        assert len(lines[0].get(near, [])) == expected_outer
        assert len(lines[-1].get(far, [])) == 1
        # Each flush seam between adjacent panels is drawn by exactly one of the two
        for upper, lower in itertools.pairwise(lines):
            assert len(upper.get(far, [])) + len(lower.get(near, [])) == 1

    @pytest.mark.parametrize("vertical", [True, False])
    def test_seams_share_one_width(
        self, make_mock_df, score_col, rank_col, token_col, score_type_col, vertical
    ):
        """Every seam is drawn by a border edge rather than a panel's own score-axis domain, so a
        border box wider than the score axis cannot leave one divider thicker than another."""
        chart = _shift(
            make_mock_df(n=50),
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            score_type_col=score_type_col,
            show_totals=True,
            vertical=vertical,
            border_config=BorderConfig(width=3),  # wider than the default score axis
        )

        lines = [_score_side_lines(panel, score_col, vertical) for panel in _panels(chart)]
        near, far = ("top", "bottom") if vertical else ("left", "right")

        seam_widths = {
            width
            for upper, lower in itertools.pairwise(lines)
            for width in upper.get(far, []) + lower.get(near, [])
        }
        assert seam_widths == {3}

    @pytest.mark.parametrize("vertical", [True, False])
    def test_seams_take_score_axis_width_when_border_hidden(
        self, make_mock_df, score_col, rank_col, token_col, score_type_col, vertical
    ):
        """With the border box off, the seams that survive are drawn at the score axis width, so
        they match the score-axis domain that separates the two totals panels."""
        chart = _shift(
            make_mock_df(n=50),
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            score_type_col=score_type_col,
            show_totals=True,
            vertical=vertical,
            border_config=BorderConfig(width=0),
            score_axis={"width": 3},
        )

        edges = set().union(*(_border_edges(panel) for panel in _panels(chart)))
        assert {width for _, width in edges} == {3}

    @pytest.mark.parametrize("vertical", [True, False])
    def test_no_border_edges_without_totals_when_border_hidden(
        self, make_mock_df, score_col, rank_col, token_col, score_type_col, vertical
    ):
        """Without totals panels there are no seams, so a hidden border box leaves the chart with
        no border edges at all — only the two native axes."""
        chart = _shift(
            make_mock_df(n=50),
            score_col=score_col,
            rank_col=rank_col,
            token_col=token_col,
            score_type_col=score_type_col,
            show_totals=False,
            vertical=vertical,
            border_config=BorderConfig(width=0),
        )

        # layer[0] is the base group of components; any border edges follow it
        assert len(cast(alt.LayerChart, chart).layer) == 1
