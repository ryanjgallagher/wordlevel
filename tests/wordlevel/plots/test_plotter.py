from typing import cast

import altair as alt
import polars as pl
import pytest

from wordlevel.catalog.catalog import Catalog
from wordlevel.comparison.comparison import comp
from wordlevel.configs import BarColorConfig, ShiftColorConfig
from wordlevel.constants import (
    PLOT_CONTRIBUTION_COL,
    PLOT_CONTRIBUTION_TYPE_COL,
    RANK_COL,
    SCORE_COL,
)
from wordlevel.plots.plotter import PlotCacheKey, Plotter, PlotType


@pytest.fixture
def df_mock() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "word": ["the", "cat", "dog", "ran", "fast"],
            "freq1": [100, 50, 30, 20, 10],
            "freq2": [90, 60, 25, 18, 15],
        }
    )


@pytest.fixture
def catalog(df_mock) -> Catalog:
    """Catalog pre-loaded with both a non-weighted-avg ('freq_comp') and weighted-avg ('entropy_comp') comparison."""
    base = Catalog(df_mock, token_col="word", corpora=["freq1", "freq2"])

    freq = comp("freq1", "freq2").score.frequency().alias("freq_comp")
    entropy = comp("freq1", "freq2").score.tsallis_entropy().alias("entropy_comp")

    return base.with_comparisons(freq, entropy)


class TestGetPlotDf:
    @pytest.mark.parametrize("plot_type", ["bar", "shift"])
    def test_output_columns(self, catalog: Catalog, plot_type: PlotType) -> None:
        """Output DataFrame has exactly the four expected columns regardless of plot type."""
        plotter = Plotter(catalog.comparisons, "word")

        df_result = plotter._get_plot_df(
            comparison=catalog.comparisons["entropy_comp"],
            plot_type=plot_type,
            max_rank=50,
        )

        assert set(df_result.columns) == {
            "word",
            RANK_COL,
            PLOT_CONTRIBUTION_COL,
            PLOT_CONTRIBUTION_TYPE_COL,
        }

    def test_caching_same_max_rank(self, catalog: Catalog) -> None:
        """The same DataFrame object is returned on repeated calls with the same max_rank."""
        plotter = Plotter(catalog.comparisons, "word")
        comp_result = catalog.comparisons["entropy_comp"]

        df1 = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=50)
        df2 = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=50)

        assert df1 is df2
        assert isinstance(df1, pl.DataFrame)

        cached = list(plotter._plot_dfs.values())[0]  # ruff: ignore[RUF015]
        assert cached.max_rank == 50

    def test_caching_smaller_max_rank(self, catalog: Catalog) -> None:
        """A smaller max_rank returns a filtered view from the cache without recomputing."""
        plotter = Plotter(catalog.comparisons, "word")
        comp_result = catalog.comparisons["entropy_comp"]

        df_large = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=50)
        df_small = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=2)

        assert df_small.height < df_large.height
        assert df_small is not df_large
        assert len(plotter._plot_dfs) == 1

        cached = list(plotter._plot_dfs.values())[0]  # ruff: ignore[RUF015]
        assert cached.max_rank == 50

    def test_caching_larger_max_rank_recomputes(self, catalog: Catalog) -> None:
        """A larger max_rank triggers recomputation and overwrites the cache entry."""
        plotter = Plotter(catalog.comparisons, "word")
        comp_result = catalog.comparisons["entropy_comp"]

        df_small = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=2)
        df_large = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=50)

        assert df_large.height > df_small.height
        assert df_large is not df_small
        assert len(plotter._plot_dfs) == 1

        cached = list(plotter._plot_dfs.values())[0]  # ruff: ignore[RUF015]
        assert cached.max_rank == 50

    def test_caching_isolates_by_split_flag(self, catalog: Catalog) -> None:
        """Different split_nonnegative_by_corpus values produce independent cache entries."""
        plotter = Plotter(catalog.comparisons, "word")
        comp_result = catalog.comparisons["entropy_comp"]

        df_split = plotter._get_plot_df(
            comparison=comp_result, plot_type="bar", max_rank=50, split_nonnegative_by_corpus=True
        )
        df_nosplit = plotter._get_plot_df(
            comparison=comp_result, plot_type="bar", max_rank=50, split_nonnegative_by_corpus=False
        )

        assert df_split is not df_nosplit
        assert len(plotter._plot_dfs) == 2

    def test_show_totals_fetches_full_vocab_and_caches_separately(self, catalog: Catalog) -> None:
        """`show_totals` keeps the full vocabulary (so totals sum over every token) and is cached
        under a separate key from the top-`max_rank` (totals-hidden) df."""
        plotter = Plotter(catalog.comparisons, "word")
        comp_result = catalog.comparisons["entropy_comp"]

        df_top2 = plotter._get_plot_df(
            comparison=comp_result, plot_type="shift", max_rank=2, show_totals=False
        )
        df_totals = plotter._get_plot_df(
            comparison=comp_result, plot_type="shift", max_rank=None, show_totals=True
        )

        # The catalog has 5 tokens: the totals df keeps all of them; the display df is the top 2.
        assert df_totals["word"].n_unique() == 5
        assert df_top2["word"].n_unique() == 2

        # Separate cache entries; the totals entry is unfiltered (`max_rank` is None).
        assert len(plotter._plot_dfs) == 2
        totals_key = PlotCacheKey(
            alias=comp_result.alias,
            plot_type="shift",
            split_nonnegative_by_corpus=True,
            show_totals=True,
        )
        assert plotter._plot_dfs[totals_key].max_rank is None

    def test_caching_serves_concrete_from_full_df(self, catalog: Catalog) -> None:
        """A concrete max_rank request is served (filtered) from a cached full (`None`) df without
        recomputing or overwriting the cache entry."""
        plotter = Plotter(catalog.comparisons, "word")
        comp_result = catalog.comparisons["entropy_comp"]

        df_full = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=None)
        df_top2 = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=2)

        assert df_top2["word"].n_unique() == 2
        assert df_top2.height < df_full.height
        assert len(plotter._plot_dfs) == 1

        cached = list(plotter._plot_dfs.values())[0]  # ruff: ignore[RUF015]
        assert cached.max_rank is None  # the full-df entry is unchanged


class TestBorrowSymbol:
    @pytest.fixture
    def borrow_catalog(self, df_mock: pl.DataFrame) -> Catalog:
        """Catalog whose lexicon comparison borrows scores for "ran" and "fast"."""
        borrowed = (
            comp("freq1", "freq2")
            .score.lexicon(
                lexicon_reference={"the": 1.0, "cat": 5.0, "dog": 3.0, "ran": 2.0},  # no "fast"
                lexicon_comparison={"the": 2.0, "cat": 4.0, "dog": 1.0, "fast": 8.0},  # no "ran"
                scored_in_one_lexicon="borrow",
            )
            .alias("borrow_comp")
        )

        base = Catalog(df_mock, token_col="word", corpora=["freq1", "freq2"])

        return base.with_comparisons(borrowed)

    def test_symbol_appended_to_borrowed_tokens_only(self, borrow_catalog: Catalog) -> None:
        """Tokens with a borrowed score end in the symbol; tokens in both lexicons are unchanged."""
        plotter = Plotter(borrow_catalog.comparisons, "word")

        df_result = plotter._get_plot_df(
            comparison=borrow_catalog.comparisons["borrow_comp"],
            plot_type="bar",
            max_rank=50,
            borrow_symbol="*",
        )

        assert set(df_result["word"].to_list()) == {"the", "cat", "dog", "ran*", "fast*"}

    def test_no_symbol_appended_when_symbol_is_none(self, borrow_catalog: Catalog) -> None:
        """Without a symbol, borrowed tokens keep their original labels."""
        plotter = Plotter(borrow_catalog.comparisons, "word")

        df_result = plotter._get_plot_df(
            comparison=borrow_catalog.comparisons["borrow_comp"],
            plot_type="bar",
            max_rank=50,
            borrow_symbol=None,
        )

        assert set(df_result["word"].to_list()) == {"the", "cat", "dog", "ran", "fast"}

    def test_symbol_is_not_cached_with_the_plot_df(self, borrow_catalog: Catalog) -> None:
        """A later call re-labels from the cached, unsuffixed tokens rather than the labelled ones."""
        plotter = Plotter(borrow_catalog.comparisons, "word")
        comp_result = borrow_catalog.comparisons["borrow_comp"]

        plotter._get_plot_df(
            comparison=comp_result, plot_type="bar", max_rank=50, borrow_symbol="*"
        )
        df_result = plotter._get_plot_df(
            comparison=comp_result, plot_type="bar", max_rank=50, borrow_symbol="#"
        )

        # A stale cache would give "ran*"; a suffix applied to a suffixed label would give "ran*#"
        assert set(df_result["word"].to_list()) == {"the", "cat", "dog", "ran#", "fast#"}

    def test_symbol_ignored_when_the_comparison_cannot_borrow(self, catalog: Catalog) -> None:
        """A symbol is a no-op for a measure that never borrows scores, rather than an error."""
        plotter = Plotter(catalog.comparisons, "word")

        df_result = plotter._get_plot_df(
            comparison=catalog.comparisons["entropy_comp"],
            plot_type="bar",
            max_rank=50,
            borrow_symbol="*",
        )

        assert set(df_result["word"].to_list()) == {"the", "cat", "dog", "ran", "fast"}


def _grand_total_bars_layer(chart: alt.TopLevelMixin) -> alt.Chart:
    """Extracts the bars layer of the grand total panel, the middle panel of the concatenation."""
    return cast(alt.Chart, cast(alt.VConcatChart, chart).vconcat[1].layer[0].layer[0])


def _score_axis_title(chart: alt.TopLevelMixin) -> str:
    """Extracts the score (x) axis title from the bars layer of a rendered chart."""
    return cast(alt.LayerChart, chart).layer[0].layer[0].to_dict()["encoding"]["x"]["axis"]["title"]


class TestBar:
    def test_wiring(self, df_mock: pl.DataFrame, catalog: Catalog) -> None:
        """bar() returns a LayerChart with 4 layers and the correct number of data rows."""
        plotter = Plotter(catalog.comparisons, "word")
        chart = plotter.bar("freq_comp")

        assert isinstance(chart, alt.LayerChart)
        # shift_borders groups the 4 base layers in chart.layer[0] and appends border edges
        assert len(chart.layer[0].layer) == 4  # bars, labels, scales, zero line
        assert (
            len(cast(pl.DataFrame, chart.layer[0].layer[0].data)) == df_mock.height * 2
        )  # reference, comparison

    @pytest.mark.parametrize(
        "color_config",
        [
            BarColorConfig(reference="#ff0000"),
            {"reference": "#ff0000"},
        ],
    )
    def test_color_config_accepts_object_or_dict(
        self, catalog: Catalog, color_config: BarColorConfig | dict
    ) -> None:
        """`color_config` accepts either a `BarColorConfig` instance or a matching dict."""
        plotter = Plotter(catalog.comparisons, "word")
        chart = plotter.bar("freq_comp", color_config=color_config)

        # The reference color override should propagate to the bars layer's color range
        scale = chart.layer[0].layer[0].to_dict()["encoding"]["color"]["scale"]
        assert "#ff0000" in scale["range"]

    @pytest.mark.parametrize(
        "color_config",
        [
            BarColorConfig(reinforce="#00ff00"),
            {"reinforce": "#00ff00"},
        ],
    )
    def test_color_config_accepts_score_sign_colors(
        self, catalog: Catalog, color_config: BarColorConfig | dict
    ) -> None:
        """For a score-sign measure, the same `BarColorConfig` carries the reinforce/offset pair."""
        plotter = Plotter(catalog.comparisons, "word")
        chart = plotter.bar("entropy_comp", color_config=color_config)

        scale = chart.layer[0].layer[0].to_dict()["encoding"]["color"]["scale"]
        assert "#00ff00" in scale["range"]

    def test_default_score_sign_bar_uses_reinforce_offset_domain(self, catalog: Catalog) -> None:
        """A score-sign comparison colors its bars by the reinforce/offset labels, not by corpus."""
        plotter = Plotter(catalog.comparisons, "word")
        chart = plotter.bar("entropy_comp")

        scale = chart.layer[0].layer[0].to_dict()["encoding"]["color"]["scale"]
        assert set(scale["domain"]) == {"reinforce", "offset"}

    @pytest.mark.parametrize(
        "comparison,bad_config",
        [
            # Corpus-attributed measure given sign-attribution colors
            ("freq_comp", BarColorConfig(reinforce="#ffffff")),
            ("freq_comp", {"reinforce": "#ffffff"}),
            # Sign-attributed measure given corpus-attribution colors
            ("entropy_comp", BarColorConfig(reference="#ffffff")),
            ("entropy_comp", {"reference": "#ffffff"}),
        ],
    )
    def test_color_config_mismatch_raises(
        self,
        catalog: Catalog,
        comparison: str,
        bad_config: BarColorConfig | dict,
    ) -> None:
        """
        Coloring an attribution the comparison does not produce raises, naming the pair it accepts,
        whether the color is set on a `BarColorConfig` or in a dict
        """
        plotter = Plotter(catalog.comparisons, "word")
        with pytest.raises(ValueError, match="attributed as"):
            plotter.bar(comparison, color_config=bad_config)

    @pytest.mark.parametrize(
        "color_config",
        [
            BarColorConfig(grand_total="#ff0000"),
            {"grand_total": "#ff0000"},
        ],
    )
    def test_color_config_reaches_the_grand_total_bar(
        self, catalog: Catalog, color_config: BarColorConfig | dict
    ) -> None:
        """"""
        plotter = Plotter(catalog.comparisons, "word")
        chart = plotter.bar("freq_comp", show_totals=True, color_config=color_config)

        scale = _grand_total_bars_layer(chart).to_dict()["encoding"]["color"]["scale"]
        assert scale["range"] == ["#ff0000"]

    def test_show_totals_totals_span_full_vocab(
        self, df_mock: pl.DataFrame, catalog: Catalog
    ) -> None:
        """With totals shown, bar() fetches the full-vocabulary df (so the totals sum over every
        token), even when only the top `max_rank` are displayed."""
        plotter = Plotter(catalog.comparisons, "word")
        plotter.bar("freq_comp", max_rank=2, show_totals=True)

        totals_entries = [
            cached
            for key, cached in plotter._plot_dfs.items()
            if key.plot_type == "bar" and key.show_totals
        ]
        assert len(totals_entries) == 1
        assert totals_entries[0].max_rank is None
        assert totals_entries[0].df["word"].n_unique() == df_mock.height  # all 5 tokens, not top 2


class TestShift:
    def test_non_weighted_avg_raises(self, catalog: Catalog) -> None:
        """shift() raises ValueError when the comparison is not a weighted average."""
        plotter = Plotter(catalog.comparisons, "word")

        with pytest.raises(ValueError):
            plotter.shift("freq_comp")

    def test_wiring(self, df_mock: pl.DataFrame, catalog: Catalog) -> None:
        """shift() returns a LayerChart with 4 layers and the correct number of data rows."""
        plotter = Plotter(catalog.comparisons, "word")
        chart = plotter.shift("entropy_comp")

        assert isinstance(chart, alt.LayerChart)
        # shift_borders groups the 4 base layers in chart.layer[0] and appends border edges
        assert len(chart.layer[0].layer) == 4  # bars, labels, scales, zero line
        assert (
            len(cast(pl.DataFrame, chart.layer[0].layer[0].data)) == df_mock.height * 12
        )  # 12 shift types

    def test_show_totals_totals_span_full_vocab(
        self, df_mock: pl.DataFrame, catalog: Catalog
    ) -> None:
        """With totals shown, shift() fetches the full-vocabulary df (so the totals sum over every
        token), even when only the top `max_rank` are displayed."""
        plotter = Plotter(catalog.comparisons, "word")
        plotter.shift("entropy_comp", max_rank=2, show_totals=True)

        totals_entries = [
            cached
            for key, cached in plotter._plot_dfs.items()
            if key.plot_type == "shift" and key.show_totals
        ]
        assert len(totals_entries) == 1
        assert totals_entries[0].max_rank is None
        assert totals_entries[0].df["word"].n_unique() == df_mock.height  # all 5 tokens, not top 2

    @pytest.mark.parametrize(
        "color_config",
        [
            ShiftColorConfig(more_of_positive="#ff0000"),
            {"more_of_positive": "#ff0000"},
        ],
    )
    def test_color_config_accepts_object_or_dict(
        self, catalog: Catalog, color_config: ShiftColorConfig | dict
    ) -> None:
        """`color_config` accepts either a `ShiftColorConfig` instance or a matching dict."""
        plotter = Plotter(catalog.comparisons, "word")
        chart = plotter.shift("entropy_comp", color_config=color_config)

        scale = chart.layer[0].layer[0].to_dict()["encoding"]["color"]["scale"]
        assert "#ff0000" in scale["range"]

    def test_color_config_reaches_the_grand_total_bar(self, catalog: Catalog) -> None:
        """"""
        plotter = Plotter(catalog.comparisons, "word")
        chart = plotter.shift(
            "entropy_comp", show_totals=True, color_config={"grand_total": "#ff0000"}
        )

        scale = _grand_total_bars_layer(chart).to_dict()["encoding"]["color"]["scale"]
        assert scale["range"] == ["#ff0000"]


class TestNormalizedPlot:
    """Integration of normalization with plotting: percent scaling and the percent axis title."""

    @pytest.fixture
    def normalized_catalog(self, df_mock: pl.DataFrame) -> Catalog:
        return Catalog(df_mock, token_col="word", corpora=["freq1", "freq2"]).with_comparisons(
            comp("freq1", "freq2").score.tsallis_entropy().alias("raw"),
            comp("freq1", "freq2").score.tsallis_entropy().normalize("total_diff").alias("normed"),
        )

    def test_contributions_scaled_to_percent(self, normalized_catalog: Catalog) -> None:
        """Normalized plot contributions are the normalized scores multiplied by 100."""
        result = normalized_catalog.comparisons["normed"]

        df_plot = normalized_catalog.plot._get_plot_df(
            comparison=result, plot_type="bar", max_rank=50
        )

        expected = 100 * cast(float, result.df[SCORE_COL].abs().max())
        assert df_plot[PLOT_CONTRIBUTION_COL].abs().max() == pytest.approx(expected)

    def test_axis_title_reflects_normalization(self, normalized_catalog: Catalog) -> None:
        """The percent flag is wired through so a normalized plot gets the percent axis title."""
        assert _score_axis_title(normalized_catalog.plot.bar("normed")) == "Percent contribution"
        assert _score_axis_title(normalized_catalog.plot.bar("raw")) == "Contribution"
