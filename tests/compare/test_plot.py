import pytest
import altair as alt
import polars as pl

from wordlevel.compare.comparator import CorpusComparator
from wordlevel.compare.comparisons.comparison import comp
from wordlevel.compare.plot import Plotter
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
def comparator(df_mock) -> CorpusComparator:
    """Comparator pre-loaded with both a non-weighted-avg ('freq_comp') and weighted-avg ('entropy_comp') comparison."""
    base = CorpusComparator(df_mock, label_col="word", freq_cols=["freq1", "freq2"])

    freq = comp("freq1", "freq2").score.frequency().alias("freq_comp")
    entropy = comp("freq1", "freq2").score.tsallis_entropy().alias("entropy_comp")

    return base.with_comparisons(freq, entropy)


class TestGetPlotDf:
    @pytest.mark.parametrize("plot_type", ["bar", "shift"])
    def test_output_columns(self, comparator: CorpusComparator, plot_type: str) -> None:
        """Output DataFrame has exactly the four expected columns regardless of plot type."""
        plotter = Plotter(comparator)

        df_result = plotter._get_plot_df(
            comparison=comparator.comparisons["entropy_comp"],
            plot_type=plot_type,
            max_rank=50,
        )

        assert set(df_result.columns) == {
            "word",
            RANK_COL,
            PLOT_CONTRIBUTION_COL,
            PLOT_CONTRIBUTION_TYPE_COL,
        }

    def test_caching_same_max_rank(self, comparator: CorpusComparator) -> None:
        """The same DataFrame object is returned on repeated calls with the same max_rank."""
        plotter = Plotter(comparator)
        comp_result = comparator.comparisons["entropy_comp"]

        df1 = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=50)
        df2 = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=50)

        assert df1 is df2
        assert isinstance(df1, pl.DataFrame)

        cached = list(plotter._plot_dfs.values())[0]
        assert cached.max_rank == 50

    def test_caching_smaller_max_rank(self, comparator: CorpusComparator) -> None:
        """A smaller max_rank returns a filtered view from the cache without recomputing."""
        plotter = Plotter(comparator)
        comp_result = comparator.comparisons["entropy_comp"]

        df_large = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=50)
        df_small = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=2)

        assert df_small.height < df_large.height
        assert df_small is not df_large
        assert len(plotter._plot_dfs) == 1

        cached = list(plotter._plot_dfs.values())[0]
        assert cached.max_rank == 50

    def test_caching_larger_max_rank_recomputes(self, comparator: CorpusComparator) -> None:
        """A larger max_rank triggers recomputation and overwrites the cache entry."""
        plotter = Plotter(comparator)
        comp_result = comparator.comparisons["entropy_comp"]

        df_small = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=2)
        df_large = plotter._get_plot_df(comparison=comp_result, plot_type="bar", max_rank=50)

        assert df_large.height > df_small.height
        assert df_large is not df_small
        assert len(plotter._plot_dfs) == 1

        cached = list(plotter._plot_dfs.values())[0]
        assert cached.max_rank == 50

    def test_caching_isolates_by_split_flag(self, comparator: CorpusComparator) -> None:
        """Different split_nonnegative_by_corpus values produce independent cache entries."""
        plotter = Plotter(comparator)
        comp_result = comparator.comparisons["entropy_comp"]

        df_split = plotter._get_plot_df(
            comparison=comp_result, plot_type="bar", max_rank=50, split_nonnegative_by_corpus=True
        )
        df_nosplit = plotter._get_plot_df(
            comparison=comp_result, plot_type="bar", max_rank=50, split_nonnegative_by_corpus=False
        )

        assert df_split is not df_nosplit
        assert len(plotter._plot_dfs) == 2


class TestBar:
    def test_wiring(self, df_mock: pl.DataFrame, comparator: CorpusComparator) -> None:
        """bar() returns a LayerChart with 3 layers and the correct number of data rows."""
        plotter = Plotter(comparator)
        chart = plotter.bar("freq_comp")

        assert isinstance(chart, alt.LayerChart)
        assert len(chart.layer) == 3
        assert len(chart.layer[0].data) == df_mock.height * 2  # reference, comparison


class TestShift:
    def test_non_weighted_avg_raises(self, comparator: CorpusComparator) -> None:
        """shift() raises ValueError when the comparison is not a weighted average."""
        plotter = Plotter(comparator)

        with pytest.raises(ValueError):
            plotter.shift("freq_comp")

    def test_wiring(self, df_mock: pl.DataFrame, comparator: CorpusComparator) -> None:
        """shift() returns a LayerChart with 3 layers and the correct number of data rows."""
        plotter = Plotter(comparator)
        chart = plotter.shift("entropy_comp")

        assert isinstance(chart, alt.LayerChart)
        assert len(chart.layer) == 3
        assert len(chart.layer[0].data) == df_mock.height * 12  # 12 shift types
