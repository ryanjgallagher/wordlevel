import polars as pl
import pytest
from polars.testing import assert_frame_equal

from wordlevel.api.comparison.comparison import Comparison, ComparisonResult, comp
from wordlevel.constants import FREQ_CMP_COL, FREQ_REF_COL, NORMALIZATION_CONSTANT_COL


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def df_mock() -> pl.DataFrame:
    """Simple test DataFrame with reference and comparison columns."""
    return pl.DataFrame(
        {
            "word": ["the", "cat", "dog"],
            "ref": [100, 50, 25],
            "cmp": [20, 30, 80],
        }
    )


@pytest.fixture
def simple_plan():
    """Basic scoring plan that adds a comparison_score column."""

    def plan(df: pl.DataFrame | pl.LazyFrame, token_col: str) -> pl.DataFrame | pl.LazyFrame:
        return df.with_columns((pl.col(FREQ_REF_COL) - pl.col(FREQ_CMP_COL)).alias("difference"))

    return plan


@pytest.fixture
def valid_comparison(simple_plan) -> Comparison:
    """Comparison with all required fields populated."""
    return Comparison("ref", "cmp")._update_state(
        alias="test_alias",
        method="test_method",
        is_weighted_avg=False,
        is_nonnegative=False,
        attributed_by="frequency",
        score_plan=simple_plan,
        parameters={"alpha": 1.0},
    )


# ============================================================================
# TestInitialization - Test direct construction
# ============================================================================
@pytest.mark.parametrize(
    "method,is_weighted_avg,is_nonnegative,attributed_by,parameters",
    [
        ("rank_divergence", False, True, "rank", {"alpha": 1.0}),
        ("proportion", False, False, "frequency", {}),
        ("jensen_shannon_divergence", True, True, "frequency", {"alpha": 1.0, "base": 2}),
    ],
)
def test_initialization(
    df_mock, method, is_weighted_avg, is_nonnegative, attributed_by, parameters
) -> None:
    """Test initialization with different scoring methods and parameters."""
    result = ComparisonResult(
        _lf=df_mock.lazy(),
        reference="ref",
        comparison="cmp",
        alias=f"{method}_test",
        method=method,
        is_weighted_avg=is_weighted_avg,
        is_nonnegative=is_nonnegative,
        attributed_by=attributed_by,
        parameters=parameters,
    )

    assert result.reference == "ref"
    assert result.comparison == "cmp"
    assert result.alias == f"{method}_test"
    assert result.method == method
    assert result.is_weighted_avg == is_weighted_avg
    assert result.is_nonnegative == is_nonnegative
    assert result.attributed_by == attributed_by
    assert result.parameters == parameters
    assert isinstance(result._lf, pl.LazyFrame)


@pytest.mark.parametrize(
    "attributed_by,expected_labels",
    [
        ("frequency", ("reference", "comparison")),
        ("rank", ("reference", "comparison")),
        ("score_sign", ("reinforce", "offset")),
    ],
)
def test_attribution_labels(df_mock, attributed_by, expected_labels) -> None:
    """`attribution_labels` exposes the label pair for the comparison's attribution type."""
    result = ComparisonResult(
        _lf=df_mock.lazy(),
        reference="ref",
        comparison="cmp",
        alias="test",
        method="test",
        is_weighted_avg=False,
        is_nonnegative=False,
        attributed_by=attributed_by,
        parameters={},
    )

    assert result.attribution_labels == expected_labels


# ============================================================================
# TestFromComparison - Test factory method
# ============================================================================


class TestFromComparison:
    """Test the from_comparison() factory method."""

    @pytest.mark.parametrize("is_lazy", [True, False])
    def test_from_comparison(self, valid_comparison, df_mock, is_lazy) -> None:
        """Test successful construction from a valid Comparison."""
        if is_lazy:
            df_mock = df_mock.lazy()

        result = ComparisonResult.from_comparison(valid_comparison, df_mock, token_col="word")

        state = valid_comparison._state

        assert result.reference == state.reference
        assert result.comparison == state.comparison
        assert result.alias == state.alias
        assert result.method == state.method
        assert result.is_weighted_avg is state.is_weighted_avg
        assert result.is_nonnegative is state.is_nonnegative
        assert result.attributed_by == state.attributed_by
        assert result.parameters == state.parameters

    @pytest.mark.parametrize(
        "missing_field,error_msg",
        [
            ("score_plan", "Cannot resolve the plan of a comparison that has not been scored"),
            ("alias", "Cannot construct a comparison result without an alias"),
            ("method", "Missing comparison metadata"),
            ("is_weighted_avg", "Missing comparison metadata"),
            ("is_nonnegative", "Missing comparison metadata"),
            ("attributed_by", "Missing comparison metadata"),
        ],
    )
    def test_from_comparison_validation_errors(
        self, valid_comparison, df_mock, missing_field, error_msg
    ) -> None:
        """Test that ValueError is raised when required fields are missing."""
        comparison = valid_comparison._update_state(**{missing_field: None})

        with pytest.raises(ValueError, match=error_msg):
            ComparisonResult.from_comparison(comparison, df_mock, token_col="word")


# ============================================================================
# TestLazyEvaluation - Test lazy loading and caching
# ============================================================================


class TestLazyEvaluation:
    """Test lazy evaluation and caching behavior of the .df property."""

    def test_df_returns_dataframe(self, valid_comparison, df_mock) -> None:
        """Test that .df property returns a DataFrame, not LazyFrame."""
        result = ComparisonResult.from_comparison(valid_comparison, df_mock, token_col="word")
        df = result.df

        assert isinstance(df, pl.DataFrame)
        assert not isinstance(df, pl.LazyFrame)

    def test_df_multiple_accesses_return_equivalent_data(self, valid_comparison, df_mock) -> None:
        """Test that multiple .df accesses return equivalent DataFrames."""
        result = ComparisonResult.from_comparison(valid_comparison, df_mock, token_col="word")

        df1 = result.df
        df2 = result.df
        df3 = result.df

        # All should contain the same data (though may be different objects)
        assert_frame_equal(df1, df2)
        assert_frame_equal(df2, df3)

    def test_plan_execution_is_deferred(self, df_mock) -> None:
        """Test that plan execution is deferred until .df is accessed."""
        execution_count = {"count": 0}

        def counting_plan(
            df: pl.DataFrame | pl.LazyFrame, token_col: str
        ) -> pl.DataFrame | pl.LazyFrame:
            execution_count["count"] += 1
            return df.with_columns((pl.col(FREQ_REF_COL) - pl.col(FREQ_CMP_COL)).alias("score"))

        comparison = Comparison("ref", "cmp")._update_state(
            alias="test",
            method="test",
            is_weighted_avg=False,
            is_nonnegative=False,
            attributed_by="frequency",
            score_plan=counting_plan,
            parameters=dict(),
        )

        # Creating the result should execute the plan once upon initialization
        result = ComparisonResult.from_comparison(comparison, df_mock, token_col="word")
        assert execution_count["count"] == 1

        # Accessing .df should collect, but not re-execute the plan
        _ = result.df
        assert execution_count["count"] == 1


# ============================================================================
# TestImmutability - Test frozen dataclass behavior
# ============================================================================


@pytest.mark.parametrize(
    "field_name,new_value",
    [
        ("reference", "new_ref"),
        ("comparison", "new_cmp"),
        ("alias", "new_alias"),
        ("method", "new_method"),
        ("is_weighted_avg", True),
        ("is_nonnegative", False),
        ("attributed_by", "score_sign"),
        ("parameters", {"new": "dict"}),
        ("_lf", pl.LazyFrame()),
    ],
)
def test_fields_cannot_be_reassigned(valid_comparison, df_mock, field_name, new_value) -> None:
    """Test that frozen dataclass prevents field reassignment."""
    result = ComparisonResult.from_comparison(valid_comparison, df_mock, token_col="word")

    with pytest.raises(AttributeError):
        setattr(result, field_name, new_value)


# ============================================================================
# TestParameterIndependence - Test parameter dict copying
# ============================================================================


class TestParameterIndependence:
    def test_from_comparison_parameters_are_copied(self, valid_comparison, df_mock) -> None:
        """Test that parameters dict is copied, not shared."""
        original_params = valid_comparison._state.parameters
        result = ComparisonResult.from_comparison(valid_comparison, df_mock, token_col="word")

        # Modify the original parameters
        original_params["alpha"] = 2.0
        original_params["new_key"] = "new_value"

        # ComparisonResult should have a copy, unaffected by the change
        assert "new_key" not in result.parameters
        assert result.parameters == {"alpha": 1.0}
        assert result.parameters is not original_params

    def test_parameters_are_not_shared_between_results(self, df_mock, valid_comparison) -> None:
        """Test that multiple ComparisonResults don't share parameters dict."""
        result1 = ComparisonResult.from_comparison(valid_comparison, df_mock, token_col="word")
        result2 = ComparisonResult.from_comparison(valid_comparison, df_mock, token_col="word")

        # Modify one result's parameters
        result1.parameters["alpha"] = 2.0

        # Other result should be unaffected
        assert result2.parameters == {"alpha": 1.0}
        assert result1.parameters is not result2.parameters


# ============================================================================
# TestDataFrameExecution - Test DataFrame plan execution
# ============================================================================


class TestDataFrameExecution:
    """Test that plans are executed correctly and produce expected DataFrames."""

    def test_simple_plan_execution(self, df_mock, simple_plan) -> None:
        """Test that a simple plan adds the expected column."""
        comparison = Comparison("ref", "cmp")._update_state(
            alias="test",
            method="simple",
            is_weighted_avg=False,
            is_nonnegative=False,
            attributed_by="frequency",
            score_plan=simple_plan,
            parameters=dict(),
        )

        result = ComparisonResult.from_comparison(comparison, df_mock, token_col="word")
        result_df = result.df

        assert "difference" in result_df.columns
        assert result_df["difference"].to_list() == [80, 20, -55]

    def test_complex_plan_execution(self, df_mock) -> None:
        """Test that a complex plan with multiple columns works correctly."""

        def complex_plan(
            df: pl.DataFrame | pl.LazyFrame, token_col: str
        ) -> pl.DataFrame | pl.LazyFrame:
            diff_col = pl.col(FREQ_REF_COL) - pl.col(FREQ_CMP_COL)
            return df.with_columns(
                [
                    diff_col.alias("comparison_score"),
                    ((pl.col(FREQ_REF_COL) + pl.col(FREQ_CMP_COL)) / 2).alias("average"),
                    diff_col.rank().alias("rank"),
                ]
            )

        comparison = Comparison("ref", "cmp")._update_state(
            alias="test",
            method="complex",
            is_weighted_avg=False,
            is_nonnegative=False,
            attributed_by="frequency",
            score_plan=complex_plan,
            parameters=dict(),
        )

        result = ComparisonResult.from_comparison(comparison, df_mock, token_col="word")
        result_df = result.df

        assert "comparison_score" in result_df.columns
        assert "average" in result_df.columns
        assert "rank" in result_df.columns

        assert result_df["comparison_score"].to_list() == [80, 20, -55]
        assert result_df["average"].to_list() == [60.0, 40.0, 52.5]


# ============================================================================
# TestTokenColWiring - Test that token_col is passed through correctly to the plan
# ============================================================================


class TestTokenColWiring:
    """Test that token_col is correctly forwarded to the plan."""

    @pytest.mark.parametrize("token_col", ["word", "token", "term"])
    def test_label_col_is_forwarded_to_plan(self, df_mock, token_col) -> None:
        """Test that token_col is forwarded to the plan at execution time."""
        received = {}

        def capturing_plan(df: pl.DataFrame | pl.LazyFrame, lc: str) -> pl.DataFrame | pl.LazyFrame:
            received["token_col"] = lc
            return df

        comparison = Comparison("ref", "cmp")._update_state(
            alias="test",
            method="test",
            is_weighted_avg=False,
            is_nonnegative=False,
            attributed_by="frequency",
            score_plan=capturing_plan,
            parameters=dict(),
        )

        ComparisonResult.from_comparison(
            comparison, df_mock.rename({"word": token_col}), token_col=token_col
        )

        assert received["token_col"] == token_col


# ============================================================================
# TestNormalizationFields - Test the normalization attributes
# ============================================================================


class TestNormalizationFields:
    """Test the `normalized`, `normalized_by`, and `normalization_constant` attributes."""

    def test_unnormalized_defaults(self, df_mock) -> None:
        """An unnormalized result reports no normalization and a `None` constant."""
        result = ComparisonResult(
            _lf=df_mock.lazy(),
            reference="ref",
            comparison="cmp",
            alias="test",
            method="test",
            is_weighted_avg=False,
            is_nonnegative=False,
            attributed_by="frequency",
            parameters={},
        )

        assert result.normalized is False
        assert result.normalized_by is None
        assert result.normalization_constant is None

    def test_normalized_fields_read_from_constant_column(self, df_mock) -> None:
        """A normalized result reports its strategy and reads the constant from the frame."""
        lf = df_mock.with_columns(pl.lit(4.0).alias(NORMALIZATION_CONSTANT_COL)).lazy()

        result = ComparisonResult(
            _lf=lf,
            reference="ref",
            comparison="cmp",
            alias="test",
            method="test",
            is_weighted_avg=False,
            is_nonnegative=False,
            attributed_by="frequency",
            parameters={},
            normalized_by="sum_abs",
        )

        assert result.normalized is True
        assert result.normalized_by == "sum_abs"
        assert result.normalization_constant == 4.0

    def test_from_comparison_threads_normalization(self, df_mock) -> None:
        """`from_comparison` carries the normalization strategy and computes the constant."""
        comparison = comp("ref", "cmp").score.proportion().normalize("sum_abs").alias("test")

        result = ComparisonResult.from_comparison(comparison, df_mock, token_col="word")

        assert result.normalized_by == "sum_abs"
        assert result.normalization_constant is not None
