import pytest
import altair as alt
import polars as pl
from polars.testing import assert_frame_equal
from types import MappingProxyType

from wordlevel.constants import SCORE_COL, RANK_COL
from wordlevel.api.catalog.catalog import Catalog
from wordlevel.api.comparison.comparison import comp
from wordlevel.api.catalog.plot import Plotter


@pytest.fixture
def df_mock() -> pl.DataFrame:
    """Simple test DataFrame with multiple frequency columns."""
    return pl.DataFrame(
        {
            "word": ["the", "cat", "dog", "ran", "fast"],
            "freq1": [100, 50, 30, 20, 10],
            "freq2": [90, 60, 25, 18, 15],
            "freq3": [85, 70, 20, 15, 20],
        }
    )


@pytest.fixture
def catalog(df_mock) -> Catalog:
    """Pre-initialized Catalog for testing."""
    return Catalog(df_mock, token_col="word", corpora=["freq1", "freq2", "freq3"])


# ============================================================================
# Test adding comparisons to catalog
# ============================================================================


class TestWithComparisons:
    """Test the with_comparisons() method."""

    def test_with_comparisons_adds_single_comparison(self, catalog) -> None:
        """Test adding a single comparison returns new catalog with result."""
        comparison = comp("freq1", "freq2").score.proportion().alias("test_comp")
        catalog_new = catalog.with_comparisons(comparison)

        assert catalog_new is not catalog

        assert "test_comp" in catalog_new.comparisons
        assert "test_comp" not in catalog.comparisons

    def test_with_comparisons_adds_multiple_comparisons_immutably(self, catalog) -> None:
        """Test adding multiple comparisons in one call."""
        comp1 = comp("freq1", "freq2").score.proportion().alias("comp1")
        comp2 = comp("freq2", "freq3").score.proportion().alias("comp2")

        catalog_new = catalog.with_comparisons(comp1, comp2)

        assert "comp1" in catalog_new.comparisons
        assert "comp2" in catalog_new.comparisons
        assert len(catalog_new.comparisons) == 2
        assert len(catalog.comparisons) == 0

    def test_with_comparisons_chaining(self, catalog) -> None:
        """Test that comparisons can be chained."""
        comp1 = comp("freq1", "freq2").score.proportion().alias("comp1")
        comp2 = comp("freq2", "freq3").score.proportion().alias("comp2")
        comp3 = comp("freq1", "freq3").score.proportion().alias("comp3")

        catalog_new = (
            catalog.with_comparisons(comp1).with_comparisons(comp2).with_comparisons(comp3)
        )

        # All three comparisons present
        assert "comp1" in catalog_new.comparisons
        assert "comp2" in catalog_new.comparisons
        assert "comp3" in catalog_new.comparisons
        assert len(catalog_new.comparisons) == 3


# ============================================================================
# Test accessing properties of the catalog
# ============================================================================


class TestCatalogProperties:
    """Test the comparisons property."""

    def test_comparisons_readonly(self, catalog) -> None:
        """Test that comparisons property returns read-only MappingProxyType."""
        comparison = comp("freq1", "freq2").score.proportion().alias("test")
        catalog_new = catalog.with_comparisons(comparison)

        assert isinstance(catalog_new.comparisons, MappingProxyType)

        # Check it cannot be modified
        with pytest.raises(TypeError):
            catalog_new.comparisons["new_key"] = "value"

    def test_df_property(self, df_mock, catalog) -> None:
        """Test that df property returns DataFrame with preserved data."""
        # Returns DataFrame, not LazyFrame
        df1 = catalog.df
        assert isinstance(df1, pl.DataFrame)
        assert not isinstance(df1, pl.LazyFrame)

        # All original columns present (plus normalized versions)
        for col in df_mock.columns:
            assert col in df1.columns

        # Data integrity for original columns
        assert_frame_equal(df1.select(df_mock.columns), df_mock)

        # Multiple accesses return equivalent data
        df2 = catalog.df
        assert_frame_equal(df1, df2)

    def test_property_injection(self, df_mock) -> None:
        """Test that comparisons use catalog's token column and normalization suffix."""
        df_renamed = df_mock.rename({"word": "token"})
        catalog = Catalog(
            df_renamed,
            token_col="token",
            corpora=["freq1", "freq2"],
        )

        catalog_new = catalog.with_comparisons(
            comp("freq1", "freq2").score.proportion().alias("test_injection")
        )

        # Verify the catalog's DataFrame has both raw and normalized columns
        df = catalog_new.df
        assert "token" in df.columns
        assert "freq1__wl_normed" in df.columns
        assert "freq2__wl_normed" in df.columns

        # Verify the comparison result uses the catalog's token and normalization suffix
        result_df = catalog_new.comparisons["test_injection"].df
        assert "token" in result_df.columns
        assert "freq1__wl_normed" in result_df.columns
        assert "freq2__wl_normed" in result_df.columns

    def test_comparisons_normalize_an_unnormalized_catalog(self, df_mock) -> None:
        """
        A comparison normalizes the frequencies it scores with, so `normalize=False` only opts out
        of normalizing the catalog itself.
        """
        catalog = Catalog.from_polars(
            df_mock, token_col="word", corpora=["freq1", "freq2"], normalize=False
        ).with_comparisons(
            comp("freq1", "freq2").score.frequency().alias("raw"),
            comp("freq1", "freq2").score.proportion().alias("normalized"),
        )

        assert "freq1__wl_normed" not in catalog.df.columns

        raw = catalog.comparisons["raw"].df[SCORE_COL]
        normalized = catalog.comparisons["normalized"].df[SCORE_COL]

        assert normalized.abs().max() < 1
        assert raw.abs().max() > 1


# ============================================================================
# TestIntegration - Test end-to-end workflows
# ============================================================================


class TestIntegration:
    """Test complete end-to-end workflows."""

    def test_full_workflow(self, df_mock) -> None:
        """Test complete workflow: create catalog -> add comparisons -> access results."""
        # 1. Create catalog
        catalog = Catalog(df_mock, token_col="word", corpora=["freq1", "freq2", "freq3"])

        # 2. Add comparisons
        catalog_new = catalog.with_comparisons(
            comp("freq1", "freq2").score.proportion().alias("1vs2"),
            comp("freq2", "freq3").score.proportion().alias("2vs3"),
        )

        # 3. Access comparison results
        assert "1vs2" in catalog_new.comparisons
        assert "2vs3" in catalog_new.comparisons

        # 4. Get scored DataFrames
        df1 = catalog_new.comparisons["1vs2"].df
        df2 = catalog_new.comparisons["2vs3"].df

        # 5. Verify scores exist and are valid
        assert SCORE_COL in df1.columns
        assert SCORE_COL in df2.columns
        assert RANK_COL in df1.columns
        assert RANK_COL in df2.columns

        # 6. Verify original data is preserved in the catalog itself
        assert_frame_equal(catalog_new.df.select(df_mock.columns), df_mock)

        # 7. Verify plotting works end-to-end
        chart = catalog_new.plot.bar("1vs2")
        assert isinstance(chart, alt.LayerChart)

    def test_workflow_with_jsd(self, df_mock) -> None:
        """Test workflow with Jensen-Shannon divergence scoring."""
        catalog = Catalog(df_mock, token_col="word", corpora=["freq1", "freq2"])

        catalog_new = catalog.with_comparisons(
            comp("freq1", "freq2").score.jensen_shannon_divergence().alias("jsd")
        )

        # Access result
        result = catalog_new.comparisons["jsd"]
        df_result = result.df

        # Verify JSD-specific columns exist
        assert SCORE_COL in df_result.columns
        assert RANK_COL in df_result.columns

        # Verify metadata
        assert result.method == "jensen_shannon_divergence"
        assert result.is_weighted_avg is True

    def test_workflow_chained(self, df_mock) -> None:
        """Test workflow with chained comparison additions."""
        cl = Catalog(df_mock, token_col="word", corpora=["freq1", "freq2", "freq3"])

        catalog_new = (
            cl.with_comparisons(comp("freq1", "freq2").score.proportion().alias("1vs2"))
            .with_comparisons(comp("freq2", "freq3").score.proportion().alias("2vs3"))
            .with_comparisons(comp("freq1", "freq3").score.proportion().alias("1vs3"))
        )

        # All three comparisons present
        assert len(catalog_new.comparisons) == 3
        assert "1vs2" in catalog_new.comparisons
        assert "2vs3" in catalog_new.comparisons
        assert "1vs3" in catalog_new.comparisons


# ============================================================================
# TestPlotWiring - Test that the plot property is correctly wired
# ============================================================================


class TestPlotWiring:
    """Test that the catalog.plot cached property is correctly wired to Plotter."""

    def test_plot_property_returns_plotter(self, catalog) -> None:
        """catalog.plot returns a Plotter instance."""
        assert isinstance(catalog.plot, Plotter)

    def test_plot_property_is_cached(self, catalog) -> None:
        """catalog.plot returns the same object on repeated access."""
        assert catalog.plot is catalog.plot
