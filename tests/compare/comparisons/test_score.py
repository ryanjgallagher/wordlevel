import polars as pl
import pytest
from polars import col
from polars.testing import assert_frame_equal, assert_frame_not_equal

from wordlevel.compare.comparisons.comparison import Comparison
from wordlevel.compare.comparisons.score import Scorer
from wordlevel.expressions.utils import normalize_freqs


@pytest.fixture
def df_mock() -> pl.DataFrame:
    """Create a mock DataFrame for testing plan execution."""
    df = pl.DataFrame(
        {
            "word": ["the", "cat", "dog"],
            "ref": [100, 50, 25],
            "cmp": [20, 30, 80],
        }
    ).with_columns(
        normalize_freqs(col("ref")).alias("ref__wl_normed"),
        normalize_freqs(col("cmp")).alias("cmp__wl_normed"),
    )

    return df


def test_accessing_scorer_via_comparison() -> None:
    """Verify comparison.score returns Scorer with correct reference."""
    comparison = Comparison("ref", "cmp")

    assert isinstance(comparison.score, Scorer)
    assert comparison.score._comparison is comparison


class TestScoreFunctionWiring:
    @pytest.mark.parametrize(
        "score_func,expected_method,expected_params,expected_weighted_avg",
        [
            (lambda s: s.rank_divergence(), "rank_divergence", ["alpha"], False),
            (lambda s: s.frequency(), "frequency", [], False),
            (lambda s: s.proportion(), "proportion", ["reference_score"], True),
            (
                lambda s: s.tsallis_entropy(),
                "tsallis_entropy",
                ["alpha", "base", "reference_score"],
                True,
            ),
            (
                lambda s: s.kullback_leibler_divergence(),
                "kullback_leibler_divergence",
                ["base", "reference_score"],
                True,
            ),
            (
                lambda s: s.jensen_shannon_divergence(),
                "jensen_shannon_divergence",
                ["alpha", "base", "weight_ref", "weight_cmp", "reference_score"],
                True,
            ),
        ],
    )
    def test_score_function_metadata(
        self, score_func, expected_method, expected_params, expected_weighted_avg
    ) -> None:
        """All scoring methods return properly structured Comparison objects with correct metadata."""
        corpus1 = "freq_1990"
        corpus2 = "freq_2000"

        comparison = Comparison(corpus1, corpus2, normalization_suffix="_normed")
        result = score_func(comparison.score)

        # Returns new Comparison instance
        assert isinstance(result, Comparison)
        assert result is not comparison

        # Preserves original fields
        assert result._ref == corpus1
        assert result._cmp == corpus2
        assert result._normalization_suffix == "_normed"

        # Sets correct metadata fields
        assert result._method == expected_method
        assert result._is_weighted_avg == expected_weighted_avg
        for param in expected_params:
            assert param in result._parameters

        # Sets a default alias
        expected_alias = f"reference:{corpus1}_comparison:{corpus2}_method:{expected_method}"
        assert result._alias == expected_alias

        # Creates callable plan
        assert result._plan is not None
        assert callable(result._plan)

    @pytest.mark.parametrize(
        "score_func",
        [
            lambda s: s.rank_divergence(),
            lambda s: s.frequency(),
        ],
    )
    def test_simple_score_function_output(self, df_mock, score_func):
        """Simple scoring methods add comparison_score and rank columns."""
        comparison = Comparison("ref", "cmp", normalization_suffix="__wl_normed")

        result = score_func(comparison.score)
        df_result = result._plan(df_mock)

        assert "comparison_score" in df_result.columns
        assert "rank" in df_result.columns

        check_expr = None
        if result._method == "frequency":
            check_expr = (col("cmp") >= col("ref")) == (col("comparison_score") >= 0)

        if check_expr is not None:
            assert df_result.select(check_expr.all()).item()

    @pytest.mark.parametrize(
        "score_func",
        [
            lambda s: s.proportion(),
            lambda s: s.tsallis_entropy(),
            lambda s: s.kullback_leibler_divergence(),
            lambda s: s.jensen_shannon_divergence(),
        ],
    )
    def test_weighted_avg_score_function_output(self, df_mock, score_func):
        """Weighted average methods add shift components and plot component columns."""
        comparison = Comparison("ref", "cmp", normalization_suffix="__wl_normed")
        result = score_func(comparison.score)

        df_result = result._plan(df_mock)

        # Score columns
        assert "score_ref" in df_result.columns
        assert "score_cmp" in df_result.columns
        # Clean score columns should be dropped
        assert "raw_score_ref" not in df_result.columns
        assert "raw_score_cmp" not in df_result.columns
        assert "clean_score_ref" not in df_result.columns
        assert "clean_score_cmp" not in df_result.columns

        # Basic shift components
        assert "freq_diff" in df_result.columns
        assert "weighted_score_diff" in df_result.columns
        assert "ref_score_diff" in df_result.columns
        assert "comparison_score" in df_result.columns

        # Shift plot components
        assert "more_of_positive" in df_result.columns
        assert "less_of_negative" in df_result.columns
        assert "more_of_negative" in df_result.columns
        assert "less_of_positive" in df_result.columns
        assert "higher_score" in df_result.columns
        assert "lower_score" in df_result.columns
        assert "more_of_positive_counteracted" in df_result.columns
        assert "less_of_negative_counteracted" in df_result.columns
        assert "more_of_negative_counteracted" in df_result.columns
        assert "less_of_positive_counteracted" in df_result.columns
        assert "higher_score_counteracted" in df_result.columns
        assert "lower_score_counteracted" in df_result.columns

        # Check the reference and comparison were passed correctly. Where the (normalized) freqs
        # are larger in the comparison, we should see the same in the `freq_diff` column
        check_exprs = []
        if result._method != "kullback_leibler_divergence":
            check_exprs.append(
                (col("cmp__wl_normed") >= col("ref__wl_normed")) == (col("freq_diff") >= 0)
            )
        else:
            check_exprs.append(col("freq_diff") == 0)
        # When the normalized freqs are lager in the comparison, we should also see certain patterns
        # in the `weighted_score_diff` depending on the measure
        if result._method in {"kullback_leibler_divergence", "tsallis_entropy"}:
            check_exprs.append(
                (col("cmp__wl_normed") >= col("ref__wl_normed"))
                == (col("weighted_score_diff") <= 0)
            )
        elif result._method == "proportion":
            check_exprs.append(col("weighted_score_diff") == 0)

        for check_expr in check_exprs:
            assert df_result.select(check_expr.all()).item()

    @pytest.mark.parametrize(
        "score_func,parameter,value1,value2",
        [
            (lambda s, params: s.rank_divergence(**params), "alpha", 1, 2),
            (lambda s, params: s.tsallis_entropy(**params), "alpha", 1, 2),
            (lambda s, params: s.tsallis_entropy(**params), "base", 2, 3),
            (lambda s, params: s.kullback_leibler_divergence(**params), "base", 2, 3),
            (lambda s, params: s.jensen_shannon_divergence(**params), "alpha", 1, 2),
            (lambda s, params: s.jensen_shannon_divergence(**params), "base", 2, 3),
            (lambda s, params: s.jensen_shannon_divergence(**params), "weight_ref", 0.5, 0.09),
            (lambda s, params: s.jensen_shannon_divergence(**params), "weight_cmp", 0.5, 0.09),
        ],
    )
    def test_score_function_parameter_sensitivity(
        self, df_mock, score_func, parameter, value1, value2
    ) -> None:
        """ """
        comparison = Comparison("ref", "cmp")
        result1 = score_func(comparison.score, {parameter: value1})
        result2 = score_func(comparison.score, {parameter: value2})

        df_result1 = result1._plan(df_mock)
        df_result2 = result2._plan(df_mock)

        assert_frame_not_equal(
            df_result1, df_result2, check_row_order=False, check_column_order=False
        )

    @pytest.mark.parametrize(
        "score_func",
        [
            lambda s, ref_score: s.proportion(reference_score=ref_score),
            lambda s, ref_score: s.tsallis_entropy(reference_score=ref_score),
            lambda s, ref_score: s.kullback_leibler_divergence(reference_score=ref_score),
            lambda s, ref_score: s.jensen_shannon_divergence(reference_score=ref_score),
        ],
    )
    def test_reference_score_wiring(self, df_mock, score_func):
        """ """
        comparison = Comparison("ref", "cmp", normalization_suffix="__wl_normed")
        result_ref1 = score_func(comparison.score, 0)
        result_ref2 = score_func(comparison.score, "mean")
        result_ref3 = score_func(comparison.score, col("ref").sum())

        df_ref1 = result_ref1._plan(df_mock)
        df_ref2 = result_ref2._plan(df_mock)
        df_ref3 = result_ref3._plan(df_mock)

        # If wiring is correct, these dataframes must differ
        assert_frame_not_equal(df_ref1, df_ref2)
        assert_frame_not_equal(df_ref1, df_ref3)
        assert_frame_not_equal(df_ref2, df_ref3)


class TestAliasing:
    @pytest.mark.parametrize(
        "alias,method_name",
        [
            ("my_custom_alias", "test_method"),
            ("my_custom_alias", "jensen_shannon_divergence"),
            (None, "jensen_shannon_divergence"),
            (None, "rank_divergence"),
        ],
    )
    def test_default_comparison_aliases(self, alias, method_name):
        """Test _return_comparison uses provided alias or generates default."""
        comparison = Comparison("freq_1990", "freq_2000", alias=alias)

        result = comparison.score._return_comparison(
            plan=lambda x: x,
            method=method_name,
            parameters={"param1": "value1"},
            is_weighted_avg=False,
        )

        if alias is None:
            expected_alias = f"reference:freq_1990_comparison:freq_2000_method:{method_name}"
            assert result._alias == expected_alias
        else:
            assert result._alias == alias

    @pytest.mark.parametrize(
        "method,method_alias",
        [
            (
                lambda s: s.kullback_leibler_divergence(base=10, reference_score=5.0),
                lambda s: s.kld(base=10, reference_score=5.0),
            ),
            (
                lambda s: s.jensen_shannon_divergence(alpha=2, base=10),
                lambda s: s.jsd(alpha=2, base=10),
            ),
            (
                lambda s: s.tsallis_entropy(alpha=1, base=10, reference_score=3.0),
                lambda s: s.shannon_entropy(base=10, reference_score=3.0),
            ),
        ],
    )
    def test_method_aliases_are_equivalent(self, df_mock, method_alias, method) -> None:
        """"""
        comparison = Comparison("ref", "cmp", normalization_suffix="__wl_normed")

        method_result = method(comparison.score)
        alias_result = method_alias(comparison.score)

        # Check metadata matches
        assert alias_result._method == method_result._method
        assert alias_result._parameters == method_result._parameters
        assert alias_result._is_weighted_avg == method_result._is_weighted_avg

        # Check plans produce identical dataframes
        df_method = method_result._plan(df_mock)
        df_alias = alias_result._plan(df_mock)

        assert_frame_equal(df_method, df_alias, check_row_order=False, check_column_order=False)


class TestEdgeCases:
    @pytest.mark.parametrize(
        "score_func",
        [
            lambda s: s.rank_divergence(),
            lambda s: s.frequency(),
        ],
    )
    def test_simple_score_functions_use_raw_freq_columns(self, score_func):
        """Simple scores require raw columns - fail if only normalized columns present."""
        df_normed = pl.DataFrame(
            {
                "word": ["the", "cat", "dog"],
                "my_ref_normed": [0.5714, 0.2857, 0.1429],
                "my_cmp_normed": [0.5, 0.375, 0.125],
            }
        )

        comparison = Comparison("my_ref", "my_cmp", normalization_suffix="_normed")
        result = score_func(comparison.score)

        with pytest.raises(Exception):
            result._plan(df_normed)

    @pytest.mark.parametrize(
        "score_func",
        [
            lambda s: s.proportion(),
            lambda s: s.tsallis_entropy(),
            lambda s: s.kullback_leibler_divergence(),
            lambda s: s.jensen_shannon_divergence(),
        ],
    )
    def test_weighted_avg_score_functions_use_normed_freq_columns(self, score_func):
        """Weighted average scores require normalized columns - fail if missing."""
        df_raw = pl.DataFrame(
            {
                "word": ["the", "cat", "dog"],
                "my_ref": [100, 50, 25],
                "my_cmp": [80, 60, 20],
            }
        )

        comparison = Comparison("my_ref", "my_cmp", normalization_suffix="_normed")
        result = score_func(comparison.score)

        with pytest.raises(Exception):
            result._plan(df_raw)

    def test_rank_divergence_with_negative_alpha(self, df_mock):
        """Test that rank_divergence raises ValueError for negative alpha."""
        comparison = Comparison("ref", "cmp")
        result = comparison.score.rank_divergence(alpha=-1.0)

        # Error should be raised during plan execution
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            result._plan(df_mock)

    def test_calling_score_twice_overrides_previous(self, df_mock):
        """Calling a second scoring method on a scored Comparison overrides the first."""
        comparison = Comparison("ref", "cmp", normalization_suffix="__wl_normed")

        scored_once = comparison.score.rank_divergence()
        scored_twice = scored_once.score.jensen_shannon_divergence()

        assert scored_twice._method == "jensen_shannon_divergence"
        assert scored_twice._is_weighted_avg is True
        assert scored_twice._parameters != scored_once._parameters

        # The second plan should produce weighted-avg columns, not simple score columns
        df_result = scored_twice._plan(df_mock)
        assert "freq_diff" in df_result.columns
        assert "weighted_score_diff" in df_result.columns
