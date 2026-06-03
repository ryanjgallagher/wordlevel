import pytest
import altair as alt
import polars as pl

from wordlevel.api.catalog.catalog import Catalog
from wordlevel.api.comparison.comparison import comp
from wordlevel.api.catalog.plot import Plotter
from wordlevel.configs import BarConfig, ShiftBarConfig
from wordlevel.constants import (
    RANK_COL,
    PLOT_CONTRIBUTION_COL,
    PLOT_CONTRIBUTION_TYPE_COL,
)


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
    base = Catalog(df_mock, token_col="word", freq_cols=["freq1", "freq2"])

    freq = comp("freq1", "freq2").score.frequency().alias("freq_comp")
    entropy = comp("freq1", "freq2").score.tsallis_entropy().alias("entropy_comp")

    return base.with_comparisons(freq, entropy)


class TestGetPlotDf:
    @pytest.mark.parametrize("plot_type", ["bar", "shift"])
    def test_output_columns(self, catalog: Catalog, plot_type: str) -> None:
        """Output DataFrame has exactly the four expected columns regardless of plot type."""
        plotter = Plotter(catalog)

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
        plotter = Plotter(catalog)
        comp_result = catalog.comparisons["entropy_comp"]

        df1 = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=50)
        df2 = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=50)

        assert df1 is df2
        assert isinstance(df1, pl.DataFrame)

        cached = list(plotter._plot_dfs.values())[0]
        assert cached.max_rank == 50

    def test_caching_smaller_max_rank(self, catalog: Catalog) -> None:
        """A smaller max_rank returns a filtered view from the cache without recomputing."""
        plotter = Plotter(catalog)
        comp_result = catalog.comparisons["entropy_comp"]

        df_large = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=50)
        df_small = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=2)

        assert df_small.height < df_large.height
        assert df_small is not df_large
        assert len(plotter._plot_dfs) == 1

        cached = list(plotter._plot_dfs.values())[0]
        assert cached.max_rank == 50

    def test_caching_larger_max_rank_recomputes(self, catalog: Catalog) -> None:
        """A larger max_rank triggers recomputation and overwrites the cache entry."""
        plotter = Plotter(catalog)
        comp_result = catalog.comparisons["entropy_comp"]

        df_small = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=2)
        df_large = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=50)

        assert df_large.height > df_small.height
        assert df_large is not df_small
        assert len(plotter._plot_dfs) == 1

        cached = list(plotter._plot_dfs.values())[0]
        assert cached.max_rank == 50

    def test_caching_isolates_by_split_flag(self, catalog: Catalog) -> None:
        """Different split_nonnegative_by_corpus values produce independent cache entries."""
        plotter = Plotter(catalog)
        comp_result = catalog.comparisons["entropy_comp"]

        df_split = plotter._get_plot_df(
            comparison=comp_result, plot_type="bar", max_rank=50, split_nonnegative_by_corpus=True
        )
        df_nosplit = plotter._get_plot_df(
            comparison=comp_result, plot_type="bar", max_rank=50, split_nonnegative_by_corpus=False
        )

        assert df_split is not df_nosplit
        assert len(plotter._plot_dfs) == 2


class TestBar:
    def test_wiring(self, df_mock: pl.DataFrame, catalog: Catalog) -> None:
        """bar() returns a LayerChart with 3 layers and the correct number of data rows."""
        plotter = Plotter(catalog)
        chart = plotter.bar("freq_comp")

        assert isinstance(chart, alt.LayerChart)
        assert len(chart.layer) == 3
        assert len(chart.layer[0].data) == df_mock.height * 2  # reference, comparison

    @pytest.mark.parametrize(
        "bar_config",
        [
            BarConfig(reference="#ff0000"),
            {"reference": "#ff0000"},
        ],
    )
    def test_bar_config_accepts_object_or_dict(
        self, catalog: Catalog, bar_config: BarConfig | dict
    ) -> None:
        """`bar_config` accepts either a `BarConfig` instance or a matching dict."""
        plotter = Plotter(catalog)
        chart = plotter.bar("freq_comp", bar_config=bar_config)

        # The reference color override should propagate to the bars layer's color range
        scale = chart.layer[0].to_dict()["encoding"]["color"]["scale"]
        assert "#ff0000" in scale["range"]


class TestShift:
    def test_non_weighted_avg_raises(self, catalog: Catalog) -> None:
        """shift() raises ValueError when the comparison is not a weighted average."""
        plotter = Plotter(catalog)

        with pytest.raises(ValueError):
            plotter.shift("freq_comp")

    def test_wiring(self, df_mock: pl.DataFrame, catalog: Catalog) -> None:
        """shift() returns a LayerChart with 3 layers and the correct number of data rows."""
        plotter = Plotter(catalog)
        chart = plotter.shift("entropy_comp")

        assert isinstance(chart, alt.LayerChart)
        assert len(chart.layer) == 3
        assert len(chart.layer[0].data) == df_mock.height * 12  # 12 shift types

    @pytest.mark.parametrize(
        "bar_config",
        [
            ShiftBarConfig(more_of_positive="#ff0000"),
            {"more_of_positive": "#ff0000"},
        ],
    )
    def test_bar_config_accepts_object_or_dict(
        self, catalog: Catalog, bar_config: ShiftBarConfig | dict
    ) -> None:
        """`bar_config` accepts either a `ShiftBarConfig` instance or a matching dict."""
        plotter = Plotter(catalog)
        chart = plotter.shift("entropy_comp", bar_config=bar_config)

        scale = chart.layer[0].to_dict()["encoding"]["color"]["scale"]
        assert "#ff0000" in scale["range"]
