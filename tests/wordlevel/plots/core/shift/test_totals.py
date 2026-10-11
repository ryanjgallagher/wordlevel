from typing import cast

import altair as alt
import polars as pl
import pytest

from wordlevel.configs import BarConfig, ShiftColorConfig, TotalsConfig
from wordlevel.plots.core.shift.bars import InternalBarConfig
from wordlevel.plots.core.shift.border import InternalBorderConfig
from wordlevel.plots.core.shift.totals import (
    _ALIGNMENT_COL,
    _LABEL_COL,
    _POSITION_COL,
    _SCORE_TYPE_COL,
    _SHIFT_TYPE_SUMMARY_POSITIONS,
    shift_total,
)


def _bars_data(chart: alt.LayerChart) -> pl.DataFrame:
    """The dataframe plotted by the bars layer (the first base component under `shift_borders`)."""
    return cast(pl.DataFrame, chart.layer[0].layer[0].data)


def _bars_encoding(chart: alt.LayerChart) -> dict:
    """The bars layer's Vega-Lite encoding dict."""
    return chart.to_dict()["layer"][0]["layer"][0]["encoding"]


def _bars_mark(chart: alt.LayerChart) -> dict:
    """The bars layer's Vega-Lite mark dict."""
    return chart.to_dict()["layer"][0]["layer"][0]["mark"]


def _rows_by_type(chart: alt.LayerChart) -> dict[str, dict]:
    """The bars-layer rows keyed by score type, for per-type lookups."""
    return {row[_SCORE_TYPE_COL]: row for row in _bars_data(chart).to_dicts()}


@pytest.fixture
def type_bar_config() -> InternalBarConfig:
    """A per-score-type bar config with the six default shift colors."""
    return InternalBarConfig.from_shift_colors(None, None)


class TestStructure:
    @pytest.mark.parametrize("grand_total", [True, False])
    def test_returns_layer_chart_with_base_components(
        self, make_mock_df, score_col, score_type_col, grand_total
    ):
        """`shift_total` returns a layer chart whose first layer groups the four base components
        (bars, labels, scales, zero line), mirroring the main `shift` chart."""
        chart = shift_total(
            make_mock_df(),
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=grand_total,
            width=100,
        )

        assert isinstance(chart, alt.LayerChart)

        base = chart.layer[0]
        assert isinstance(base, alt.LayerChart)
        assert len(base.layer) == 4


class TestGrandTotal:
    def test_grand_total_sums_all_contributions(self, score_col, score_type_col):
        """The grand total is a single bar summing every contribution (counteracted included),
        labeled with the total symbol."""
        scores = [0.5, 0.2, -0.3, 0.1]

        df = pl.DataFrame(
            {
                score_type_col: [
                    "more_of_positive",
                    "more_of_positive_counteracted",
                    "higher_score",
                    "lower_score",
                ],
                score_col: scores,
            }
        )

        chart = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=True,
            width=100,
        )

        data = _bars_data(chart)
        assert data.height == 1

        row = data.to_dicts()[0]
        assert row[score_col] == pytest.approx(sum(scores))
        assert row[_POSITION_COL] == 1
        assert row[_LABEL_COL] == TotalsConfig().grand_total_label


class TestScoreTypeTotals:
    def test_type_totals_sum_per_type_including_counteracted(
        self, score_col, score_type_col, type_bar_config
    ):
        """A score type and its `_counteracted` variant fold into one bar summing both, labeled and
        positioned by the score type."""
        scores = [0.3, 0.2]
        df = pl.DataFrame(
            {
                score_type_col: ["more_of_positive", "more_of_positive_counteracted"],
                score_col: scores,
            }
        )

        chart = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=False,
            bar_config=type_bar_config,
            width=100,
        )

        rows = _rows_by_type(chart)
        assert set(rows) == {"more_of_positive"}
        assert rows["more_of_positive"][score_col] == pytest.approx(sum(scores))
        assert rows["more_of_positive"][_LABEL_COL] == TotalsConfig().more_of_positive_label
        assert (
            rows["more_of_positive"][_POSITION_COL]
            == _SHIFT_TYPE_SUMMARY_POSITIONS["more_of_positive"]
        )

    def test_higher_lower_score_dropped_when_both_zero(
        self, score_col, score_type_col, type_bar_config
    ):
        """The higher/lower-score bars are dropped when both sum to zero (e.g. a single lexicon),
        while the frequency bars are always kept."""
        df = pl.DataFrame(
            {
                score_type_col: [
                    "more_of_positive",
                    "less_of_positive",
                    "more_of_negative",
                    "less_of_negative",
                    "higher_score",
                    "lower_score",
                ],
                score_col: [0.2, -0.1, -0.05, 0.06, 0.0, 0.0],
            }
        )

        chart = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=False,
            bar_config=type_bar_config,
            width=100,
        )

        rows = _rows_by_type(chart)
        assert "higher_score" not in rows
        assert "lower_score" not in rows
        assert set(rows) == {
            "more_of_positive",
            "less_of_positive",
            "more_of_negative",
            "less_of_negative",
        }

    def test_higher_lower_score_kept_when_one_side_nonzero(
        self, score_col, score_type_col, type_bar_config
    ):
        """If either of the higher/lower-score bars is non-zero, both are kept (the empty side is
        still shown)."""
        df = pl.DataFrame(
            {
                score_type_col: ["more_of_positive", "higher_score", "lower_score"],
                score_col: [0.2, 0.03, 0.0],
            }
        )

        chart = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=False,
            bar_config=type_bar_config,
            width=100,
        )

        rows = _rows_by_type(chart)
        assert "higher_score" in rows
        assert "lower_score" in rows

    def test_alignment_sign_follows_bar_then_partner(
        self, score_col, score_type_col, type_bar_config
    ):
        """Alignment is the sign of the summed bar; a zero-but-kept bar takes the side opposite its
        non-zero partner (so `less_of_positive == 0` still faces away from `more_of_positive`)."""
        df = pl.DataFrame(
            {
                score_type_col: [
                    "more_of_positive",
                    "less_of_positive",  # zero -> opposite of the (+) partner -> negative side
                    "more_of_negative",
                    "less_of_negative",
                ],
                score_col: [0.2, 0.0, -0.05, 0.06],
            }
        )

        chart = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=False,
            bar_config=type_bar_config,
            width=100,
        )

        rows = _rows_by_type(chart)
        assert rows["more_of_positive"][_ALIGNMENT_COL] == 1.0
        assert rows["less_of_positive"][_ALIGNMENT_COL] == -1.0
        assert rows["more_of_negative"][_ALIGNMENT_COL] == -1.0
        assert rows["less_of_negative"][_ALIGNMENT_COL] == 1.0


class TestSharedScale:
    def test_grand_and_type_totals_share_score_scale(
        self, score_col, score_type_col, type_bar_config
    ):
        """The grand-total and per-type panels are built off the same union of totals, so their
        score-axis limits match — keeping the stacked panels on a common scale."""
        df = pl.DataFrame(
            {
                score_type_col: [
                    "more_of_positive",
                    "more_of_negative",
                    "higher_score",
                    "lower_score",
                ],
                score_col: [0.4, -0.25, 0.1, -0.05],
            }
        )
        grand = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=True,
            bar_config=type_bar_config,
            width=100,
        )
        per_type = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=False,
            bar_config=type_bar_config,
            width=100,
        )

        def _limits(chart: alt.LayerChart) -> tuple[float, float]:
            # The scales layer hoists its data to the group level when layered
            df_lims = cast(pl.DataFrame, chart.layer[0].layer[2].data)
            return df_lims["positive_limit"][0], df_lims["negative_limit"][0]

        grand_pos, grand_neg = _limits(grand)
        type_pos, type_neg = _limits(per_type)
        assert grand_pos == pytest.approx(type_pos)
        assert grand_neg == pytest.approx(type_neg)


class TestConfig:
    @pytest.mark.parametrize("as_dict", [False, True], ids=["object", "dict"])
    def test_grand_total_color_and_label(self, score_col, score_type_col, as_dict):
        """The grand-total bar takes its color from the color config and its label from
        `totals_config` (as object or dict)."""
        overrides = {"grand_total_label": "T"}
        totals_config = overrides if as_dict else TotalsConfig(**overrides)
        df = pl.DataFrame({score_type_col: ["more_of_positive"], score_col: [0.5]})

        chart = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=True,
            bar_config=InternalBarConfig.from_shift_colors({"grand_total": "#000000"}, None),
            totals_config=totals_config,
            width=100,
        )

        assert "#000000" in _bars_encoding(chart)["color"]["scale"]["range"]
        assert _bars_data(chart)[_LABEL_COL][0] == "T"

    def test_labels_for_absent_score_types_are_ignored(self, score_col, score_type_col):
        """`TotalsConfig` carries a label for every contribution type of either plot, so the ones
        that do not name a score type in the data are ignored rather than rejected."""
        df = pl.DataFrame({score_type_col: ["reference"], score_col: [0.5]})

        chart = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=False,
            bar_config=InternalBarConfig.from_bar_colors(None, None, ("reference", "comparison")),
            totals_config=TotalsConfig(reference_label="Roosevelt", lower_score_label="ignored"),
            width=100,
        )

        assert _rows_by_type(chart)["reference"][_LABEL_COL] == "Roosevelt"

    def test_configs_set_type_colors_and_labels(self, score_col, score_type_col):
        """A per-type bar takes its color from `bar_config` and its label from `totals_config`."""
        bar_config = InternalBarConfig.from_shift_colors(
            ShiftColorConfig(more_of_positive="#000000"), None
        )
        df = pl.DataFrame({score_type_col: ["more_of_positive"], score_col: [0.5]})

        chart = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=False,
            bar_config=bar_config,
            totals_config=TotalsConfig(more_of_positive_label="Z"),
            width=100,
        )

        assert "#000000" in _bars_encoding(chart)["color"]["scale"]["range"]
        assert _rows_by_type(chart)["more_of_positive"][_LABEL_COL] == "Z"

    @pytest.mark.parametrize("grand_total", [True, False])
    def test_bar_config_sets_outline(self, score_col, score_type_col, grand_total):
        """Without outlines set in `totals_config`, both the grand-total and per-type bars take
        the `bar_config` outline."""
        bar_config = InternalBarConfig.from_shift_colors(
            None, BarConfig(outline_color="blue", outline_width=2)
        )
        df = pl.DataFrame({score_type_col: ["more_of_positive"], score_col: [0.5]})

        chart = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=grand_total,
            bar_config=bar_config,
            width=100,
        )

        mark = _bars_mark(chart)
        assert (mark["stroke"], mark["strokeWidth"]) == ("blue", 2)

    @pytest.mark.parametrize("as_dict", [False, True], ids=["object", "dict"])
    def test_totals_config_overrides_grand_total_outline(self, score_col, score_type_col, as_dict):
        """An outline field set in `totals_config` (as object or dict) overrides the
        `bar_config` outline on the grand-total bar, and unset fields still inherit."""
        bar_config = InternalBarConfig.from_shift_colors(
            None, BarConfig(outline_color="blue", outline_width=2)
        )
        overrides = {"grand_total_outline_color": "red"}
        totals_config = overrides if as_dict else TotalsConfig(**overrides)
        df = pl.DataFrame({score_type_col: ["more_of_positive"], score_col: [0.5]})

        chart = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=True,
            bar_config=bar_config,
            totals_config=totals_config,
            width=100,
        )

        mark = _bars_mark(chart)
        assert (mark["stroke"], mark["strokeWidth"]) == ("red", 2)

    @pytest.mark.parametrize("draw_score_axis, draw_rank_axis", [(True, False), (False, True)])
    def test_border_flags_drive_axis_domains(
        self, score_col, score_type_col, type_bar_config, draw_score_axis, draw_rank_axis
    ):
        """The border config's `draw_score_axis`/`draw_rank_axis` flags drive whether the score and
        rank axis domain lines are drawn on the re-encoded bars."""
        border_config = InternalBorderConfig(
            draw_score_axis=draw_score_axis, draw_rank_axis=draw_rank_axis
        )
        df = pl.DataFrame(
            {score_type_col: ["more_of_positive", "more_of_negative"], score_col: [0.5, -0.2]}
        )

        chart = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=False,
            bar_config=type_bar_config,
            border_config=border_config,
            width=100,
        )

        encoding = _bars_encoding(chart)
        # Vertical: the score axis is x, the rank axis is y
        assert encoding["x"]["axis"]["domain"] is draw_score_axis
        assert encoding["y"]["axis"]["domain"] is draw_rank_axis


class TestOrientation:
    @pytest.mark.parametrize("vertical", [True, False])
    def test_orientation_places_score_axis_and_swaps_channels(
        self, score_col, score_type_col, type_bar_config, vertical
    ):
        """Orientation swaps which channel carries the score vs. the position, routes the score-axis
        size to width/height accordingly, and hides the score-axis tick labels."""
        size = 321
        df = pl.DataFrame(
            {score_type_col: ["more_of_positive", "more_of_negative"], score_col: [0.5, -0.2]}
        )

        chart = shift_total(
            df,
            score_col=score_col,
            score_type_col=score_type_col,
            grand_total=False,
            bar_config=type_bar_config,
            vertical=vertical,
            width=size if vertical else None,
            height=None if vertical else size,
        )

        encoding = _bars_encoding(chart)
        spec = chart.to_dict()
        score_channel = "x" if vertical else "y"
        rank_channel = "y" if vertical else "x"

        assert encoding[score_channel]["field"] == score_col
        assert encoding[rank_channel]["field"] == _POSITION_COL
        assert encoding[score_channel]["axis"]["labels"] is False
        assert spec["width" if vertical else "height"] == size
