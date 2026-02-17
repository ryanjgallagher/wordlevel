import pytest
import polars as pl
from polars.testing import assert_frame_equal
from types import MappingProxyType

from wordlevel.compare.comparator import CorpusComparator
from wordlevel.compare.comparisons.comparison import comp


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
def comparator(df_mock) -> CorpusComparator:
    """Pre-initialized CorpusComparator for testing."""
    return CorpusComparator(
        df_mock, label_col="word", freq_cols=["freq1", "freq2", "freq3"], normalize=True
    )


# ============================================================================
# Test adding comparisons to comparator
# ============================================================================


class TestWithComparisons:
    """Test the with_comparisons() method."""

    def test_with_comparisons_adds_single_comparison(self, comparator) -> None:
        """Test adding a single comparison returns new comparator with result."""
        comparison = comp("freq1", "freq2").score.proportion().alias("test_comp")
        comparator_new = comparator.with_comparisons(comparison)

        assert comparator_new is not comparator

        assert "test_comp" in comparator_new.comparisons
        assert "test_comp" not in comparator.comparisons

    def test_with_comparisons_adds_multiple_comparisons_immutably(self, comparator) -> None:
        """Test adding multiple comparisons in one call."""
        comp1 = comp("freq1", "freq2").score.proportion().alias("comp1")
        comp2 = comp("freq2", "freq3").score.proportion().alias("comp2")

        comparator_new = comparator.with_comparisons(comp1, comp2)

        assert "comp1" in comparator_new.comparisons
        assert "comp2" in comparator_new.comparisons
        assert len(comparator_new.comparisons) == 2
        assert len(comparator.comparisons) == 0

    def test_with_comparisons_chaining(self, comparator) -> None:
        """Test that comparisons can be chained."""
        comp1 = comp("freq1", "freq2").score.proportion().alias("comp1")
        comp2 = comp("freq2", "freq3").score.proportion().alias("comp2")
        comp3 = comp("freq1", "freq3").score.proportion().alias("comp3")

        comparator_new = (
            comparator.with_comparisons(comp1).with_comparisons(comp2).with_comparisons(comp3)
        )

        # All three comparisons present
        assert "comp1" in comparator_new.comparisons
        assert "comp2" in comparator_new.comparisons
        assert "comp3" in comparator_new.comparisons
        assert len(comparator_new.comparisons) == 3


# ============================================================================
# Test accessing properties of the comparator
# ============================================================================


class TestComparatorProperties:
    """Test the comparisons property."""

    def test_comparisons_readonly(self, comparator) -> None:
        """Test that comparisons property returns read-only MappingProxyType."""
        comparison = comp("freq1", "freq2").score.proportion().alias("test")
        comparator_new = comparator.with_comparisons(comparison)

        assert isinstance(comparator_new.comparisons, MappingProxyType)

        # Check it cannot be modified
        with pytest.raises(TypeError):
            comparator_new.comparisons["new_key"] = "value"

    def test_df_property(self, df_mock, comparator) -> None:
        """Test that df property returns DataFrame with preserved data."""
        # Returns DataFrame, not LazyFrame
        df1 = comparator.df
        assert isinstance(df1, pl.DataFrame)
        assert not isinstance(df1, pl.LazyFrame)

        # All original columns present (plus normalized versions)
        for col in df_mock.columns:
            assert col in df1.columns

        # Data integrity for original columns
        assert_frame_equal(df1.select(df_mock.columns), df_mock)

        # Multiple accesses return equivalent data
        df2 = comparator.df
        assert_frame_equal(df1, df2)

    def test_suffix_injection(self, df_mock) -> None:
        """Test that comparisons use comparator's normalization suffix."""
        comparator = CorpusComparator(
            df_mock,
            label_col="word",
            freq_cols=["freq1", "freq2"],
            normalize=True,
            normalization_suffix="__custom",
        )

        comparator_new = comparator.with_comparisons(comp("freq1", "freq2").score.proportion())

        # Verify normalized columns with custom suffix exist in DataFrame
        df = comparator_new.df
        assert "freq1" in df.columns
        assert "freq2" in df.columns
        assert "freq1__custom" in df.columns
        assert "freq2__custom" in df.columns

        # Verify standard suffix is not present
        assert "freq1__wl_normed" not in df.columns
        assert "freq2__wl_normed" not in df.columns


# ============================================================================
# TestIntegration - Test end-to-end workflows
# ============================================================================


class TestIntegration:
    """Test complete end-to-end workflows."""

    def test_full_workflow(self, df_mock) -> None:
        """Test complete workflow: create comparator -> add comparisons -> access results."""
        # 1. Create comparator
        comparator = CorpusComparator(
            df_mock, label_col="word", freq_cols=["freq1", "freq2", "freq3"], normalize=True
        )

        # 2. Add comparisons
        comparator_new = comparator.with_comparisons(
            comp("freq1", "freq2").score.proportion().alias("1vs2"),
            comp("freq2", "freq3").score.proportion().alias("2vs3"),
        )

        # 3. Access comparison results
        assert "1vs2" in comparator_new.comparisons
        assert "2vs3" in comparator_new.comparisons

        # 4. Get scored DataFrames
        df1 = comparator_new.comparisons["1vs2"].df
        df2 = comparator_new.comparisons["2vs3"].df

        # 5. Verify scores exist and are valid
        assert "comparison_score" in df1.columns
        assert "comparison_score" in df2.columns
        assert "rank" in df1.columns
        assert "rank" in df2.columns

        # 6. Verify original data is preserved
        assert_frame_equal(df1.select(df_mock.columns), df_mock)
        assert_frame_equal(df2.select(df_mock.columns), df_mock)

    def test_workflow_with_jsd(self, df_mock) -> None:
        """Test workflow with Jensen-Shannon divergence scoring."""
        comparator = CorpusComparator(
            df_mock, label_col="word", freq_cols=["freq1", "freq2"], normalize=True
        )

        comparator_new = comparator.with_comparisons(
            comp("freq1", "freq2").score.jensen_shannon_divergence().alias("jsd")
        )

        # Access result
        result = comparator_new.comparisons["jsd"]
        df_result = result.df

        # Verify JSD-specific columns exist
        assert "comparison_score" in df_result.columns
        assert "rank" in df_result.columns

        # Verify metadata
        assert result.method == "jensen_shannon_divergence"
        assert result.is_weighted_avg is True

    def test_workflow_chained(self, df_mock) -> None:
        """Test workflow with chained comparison additions."""
        cc = CorpusComparator(
            df_mock, label_col="word", freq_cols=["freq1", "freq2", "freq3"], normalize=True
        )

        comparator_new = (
            cc.with_comparisons(comp("freq1", "freq2").score.proportion().alias("1vs2"))
            .with_comparisons(comp("freq2", "freq3").score.proportion().alias("2vs3"))
            .with_comparisons(comp("freq1", "freq3").score.proportion().alias("1vs3"))
        )

        # All three comparisons present
        assert len(comparator_new.comparisons) == 3
        assert "1vs2" in comparator_new.comparisons
        assert "2vs3" in comparator_new.comparisons
        assert "1vs3" in comparator_new.comparisons
