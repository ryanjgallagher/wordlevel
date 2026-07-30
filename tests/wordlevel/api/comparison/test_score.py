import polars as pl
import pytest
from polars import col
from polars.testing import assert_frame_equal, assert_frame_not_equal

from wordlevel.constants import (
    REFERENCE_ATTRIBUTION,
    COMPARISON_ATTRIBUTION,
    REINFORCE_ATTRIBUTION,
    IS_BORROWED_COL,
    SCORE_COL,
)
from wordlevel.api.comparison.comparison import Comparison
from wordlevel.api.comparison.score import Scorer
from wordlevel.core.utils import normalize_freqs
from wordlevel.core.lexicons.load import Lexicon


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
        "score_func,expected_method,expected_params,expected_weighted_avg,expected_attributed_by",
        [
            (lambda s: s.rank_divergence(), "rank_divergence", ["alpha"], False, "rank"),
            (lambda s: s.frequency(), "frequency", [], False, "frequency"),
            (lambda s: s.proportion(), "proportion", ["reference_score"], True, "frequency"),
            (
                lambda s: s.tsallis_entropy(),
                "tsallis_entropy",
                ["alpha", "base", "reference_score"],
                True,
                "score_sign",
            ),
            (
                lambda s: s.kullback_leibler_divergence(),
                "kullback_leibler_divergence",
                ["base"],
                True,
                "score_sign",
            ),
            (
                lambda s: s.jensen_shannon_divergence(),
                "jensen_shannon_divergence",
                ["alpha", "base", "weight_ref", "weight_cmp", "reference_score"],
                True,
                "frequency",
            ),
            (
                lambda s: s.lexicon(lexicon={"the": 1.0}),
                "lexicon",
                ["reference_score", "impute"],
                True,
                "score_sign",
            ),
        ],
    )
    def test_score_function_metadata(
        self,
        score_func,
        expected_method,
        expected_params,
        expected_weighted_avg,
        expected_attributed_by,
    ) -> None:
        """All scoring methods return properly structured Comparison objects with correct metadata."""
        corpus1 = "freq_1990"
        corpus2 = "freq_2000"

        comparison = Comparison(corpus1, corpus2)
        result = score_func(comparison.score)

        # Returns new Comparison instance
        assert isinstance(result, Comparison)
        assert result is not comparison

        # Preserves original fields
        assert result._state.reference == corpus1
        assert result._state.comparison == corpus2

        # Sets correct metadata fields
        assert result._state.method == expected_method
        assert result._state.is_weighted_avg == expected_weighted_avg
        assert result._state.attributed_by == expected_attributed_by
        for param in expected_params:
            assert param in result._state.parameters

        # Sets a default alias
        expected_alias = f"reference:{corpus1}_comparison:{corpus2}_method:{expected_method}"
        assert result._state.alias == expected_alias

        # Creates callable plan
        assert result._state.score_plan is not None
        assert callable(result._state.score_plan)

    def test_frequency_scorer_wiring(self, df_mock):
        """Frequency scorer produces scores whose sign matches the raw frequency comparison direction."""
        comparison = Comparison("ref", "cmp")
        df_result = comparison.score.frequency()._state.resolve()(df_mock, "word", "__wl_normed")

        check_expr = (col("cmp") >= col("ref")) == (col("comparison_score") >= 0)
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
    def test_weighted_avg_scorer_wiring(self, df_mock, score_func):
        """Weighted average scorers wire frequency columns to produce correctly-directed shift components."""
        comparison = Comparison("ref", "cmp")
        result = score_func(comparison.score)
        df_result = result._state.resolve()(df_mock, "word", "__wl_normed")

        # Where the (normalized) freqs are larger in the comparison, we should see the same in
        # the `freq_diff` column (KLD is an exception: it uses only comparison freqs, so diff is 0)
        check_exprs = []
        if result._state.method != "kullback_leibler_divergence":
            check_exprs.append(
                (col("cmp__wl_normed") >= col("ref__wl_normed")) == (col("freq_diff") >= 0)
            )
        else:
            check_exprs.append(col("freq_diff") == 0)
        # When the normalized freqs are larger in the comparison, we should also see certain
        # patterns in the `weighted_score_diff` depending on the measure. KLD weights by the
        # comparison and gives positive scores for comparison-overrepresented tokens; Tsallis
        # entropy weights by each corpus independently and gives the opposite sign.
        if result._state.method == "tsallis_entropy":
            check_exprs.append(
                (col("cmp__wl_normed") >= col("ref__wl_normed"))
                == (col("weighted_score_diff") <= 0)
            )
        elif result._state.method == "kullback_leibler_divergence":
            check_exprs.append(
                (col("cmp__wl_normed") >= col("ref__wl_normed"))
                == (col("weighted_score_diff") >= 0)
            )
        elif result._state.method == "proportion":
            check_exprs.append(col("weighted_score_diff") == 0)

        for check_expr in check_exprs:
            assert df_result.select(check_expr.all()).item()

    @pytest.mark.parametrize(
        "score_func, expected_attribution",
        [
            (lambda s: s.rank_divergence(), COMPARISON_ATTRIBUTION),
            (lambda s: s.jensen_shannon_divergence(), REFERENCE_ATTRIBUTION),
            (lambda s: s.frequency(), REFERENCE_ATTRIBUTION),
            (lambda s: s.proportion(), REFERENCE_ATTRIBUTION),
        ],
    )
    def test_frequency_and_rank_attribution_wiring(self, df_mock, score_func, expected_attribution):
        """Confirm frequency and rank attributors are wired to the correct methods.

        'cat' is the discriminating word: equal rank in both corpora but higher normalized frequency
        in the reference. Rank attribution gives 'comparison' (cmp_rank <= ref_rank); frequency
        attribution gives 'reference' (ref > cmp).
        """
        comparison = Comparison("ref", "cmp")
        df_result = score_func(comparison.score)._state.resolve()(df_mock, "word", "__wl_normed")

        cat_attribution = df_result.filter(col("word") == "cat")["attribution"].item()
        assert cat_attribution == expected_attribution

    @pytest.mark.parametrize(
        "score_func",
        [
            lambda s: s.tsallis_entropy(),
            lambda s: s.kullback_leibler_divergence(),
            lambda s: s.lexicon(lexicon={"the": 1.0, "cat": 5.0, "dog": -3.0}),
        ],
    )
    def test_score_sign_attribution_wiring(self, df_mock, score_func):
        """Confirm score_sign attribution is wired to the correct methods.

        Checks the invariant: attribution follows the direction of the total shift. A contribution
        with the same sign as the total shift reinforces it; otherwise it offsets it.
        """
        comparison = Comparison("ref", "cmp")
        df_result = score_func(comparison.score)._state.resolve()(df_mock, "word", "__wl_normed")

        total_shift_pos = df_result.select(col("comparison_score").sum() >= 0).item()
        if total_shift_pos:
            check = (col("comparison_score") >= 0) == (col("attribution") == REINFORCE_ATTRIBUTION)
        else:
            check = (col("comparison_score") <= 0) == (col("attribution") == REINFORCE_ATTRIBUTION)

        assert df_result.select(check.all()).item()

    def test_kld_raises_when_comparison_has_token_absent_from_reference(self) -> None:
        """KLD is infinite when the comparison has a token the reference lacks; surface that
        up front rather than letting the framework produce silent NaN/inf results."""
        df = pl.DataFrame(
            {
                "word": ["the", "cat", "dog"],
                "ref": [5, 3, 0],  # reference lacks 'dog'
                "cmp": [1, 2, 4],  # comparison has 'dog'
            }
        ).with_columns(
            normalize_freqs(col("ref")).alias("ref__wl_normed"),
            normalize_freqs(col("cmp")).alias("cmp__wl_normed"),
        )
        comparison = Comparison("ref", "cmp")
        plan = comparison.score.kullback_leibler_divergence()._state.resolve()

        with pytest.raises(ValueError, match="Kullback-Leibler divergence is infinite"):
            plan(df, "word", "__wl_normed")

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

        df_result1 = result1._state.resolve()(df_mock, "word", "__wl_normed")
        df_result2 = result2._state.resolve()(df_mock, "word", "__wl_normed")

        assert_frame_not_equal(
            df_result1, df_result2, check_row_order=False, check_column_order=False
        )

    @pytest.mark.parametrize(
        "score_func",
        [
            lambda s, ref_score: s.proportion(reference_score=ref_score),
            lambda s, ref_score: s.tsallis_entropy(reference_score=ref_score),
            lambda s, ref_score: s.jensen_shannon_divergence(reference_score=ref_score),
            lambda s, ref_score: s.lexicon(
                lexicon={"the": 1.0, "cat": 5.0, "dog": -3.0}, reference_score=ref_score
            ),
        ],
    )
    def test_reference_score_wiring(self, df_mock, score_func):
        """ """
        comparison = Comparison("ref", "cmp")
        result_ref1 = score_func(comparison.score, 0)
        result_ref2 = score_func(comparison.score, "mean")
        result_ref3 = score_func(comparison.score, col("ref__wl_normed").pow(2).sum())

        df_ref1 = result_ref1._state.resolve()(df_mock, "word", "__wl_normed")
        df_ref2 = result_ref2._state.resolve()(df_mock, "word", "__wl_normed")
        df_ref3 = result_ref3._state.resolve()(df_mock, "word", "__wl_normed")

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
        comparison = Comparison("freq_1990", "freq_2000")._update_state(alias=alias)

        result = comparison.score._return_comparison(
            score_plan=lambda x: x,
            method=method_name,
            parameters={"param1": "value1"},
            is_weighted_avg=False,
            is_nonnegative=False,
            attributed_by="frequency",
        )

        if alias is None:
            expected_alias = f"reference:freq_1990_comparison:freq_2000_method:{method_name}"
            assert result._state.alias == expected_alias
        else:
            assert result._state.alias == alias

    @pytest.mark.parametrize(
        "method,method_alias",
        [
            (
                lambda s: s.kullback_leibler_divergence(base=10),
                lambda s: s.kld(base=10),
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
        comparison = Comparison("ref", "cmp")

        method_result = method(comparison.score)
        alias_result = method_alias(comparison.score)

        # Check metadata matches
        assert alias_result._state.method == method_result._state.method
        assert alias_result._state.parameters == method_result._state.parameters
        assert alias_result._state.is_weighted_avg == method_result._state.is_weighted_avg

        # Check plans produce identical dataframes
        df_method = method_result._state.resolve()(df_mock, "word", "__wl_normed")
        df_alias = alias_result._state.resolve()(df_mock, "word", "__wl_normed")

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

        comparison = Comparison("my_ref", "my_cmp")
        result = score_func(comparison.score)

        with pytest.raises(Exception):
            result._state.resolve()(df_normed, "word", "_normed")

    @pytest.mark.parametrize(
        "score_func",
        [
            lambda s: s.proportion(),
            lambda s: s.tsallis_entropy(),
            lambda s: s.kullback_leibler_divergence(),
            lambda s: s.jensen_shannon_divergence(),
            lambda s: s.lexicon(lexicon={"the": 1.0, "cat": 5.0, "dog": -3.0}),
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

        comparison = Comparison("my_ref", "my_cmp")
        result = score_func(comparison.score)

        with pytest.raises(Exception):
            result._state.resolve()(df_raw, "word", "_normed")

    @pytest.mark.parametrize(
        "score_func",
        [
            lambda s: s.rank_divergence(),
            lambda s: s.frequency(),
            lambda s: s.proportion(),
            lambda s: s.jensen_shannon_divergence(),
            lambda s: s.lexicon(lexicon={"the": 1.0, "cat": 1.0, "dog": 1.0}),
        ],
    )
    def test_plan_uses_provided_label_and_suffix(self, df_mock, score_func):
        """Plans use the token_col and normalization_suffix passed at execution time, not hardcoded values."""
        comparison = Comparison("ref", "cmp")
        result = score_func(comparison.score)

        df_renamed = df_mock.rename(
            {"word": "token", "ref__wl_normed": "ref_norm", "cmp__wl_normed": "cmp_norm"}
        )

        df_result_default = result._state.resolve()(df_mock, "word", "__wl_normed")
        df_result_renamed = result._state.resolve()(df_renamed, "token", "_norm")

        # The renamed result should use "token" as the token column, not "word"
        assert "token" in df_result_renamed.columns
        assert "word" not in df_result_renamed.columns

        # Scores should be identical regardless of column naming
        assert_frame_equal(
            df_result_default.sort("word").select("comparison_score"),
            df_result_renamed.sort("token").select("comparison_score"),
        )

    def test_rank_divergence_with_negative_alpha(self, df_mock):
        """Test that rank_divergence raises ValueError for negative alpha."""
        comparison = Comparison("ref", "cmp")
        result = comparison.score.rank_divergence(alpha=-1.0)

        # Error should be raised during plan execution
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            result._state.resolve()(df_mock, "word", "__wl_normed")

    def test_calling_score_twice_overrides_previous(self, df_mock):
        """Calling a second scoring method on a scored Comparison overrides the first."""
        comparison = Comparison("ref", "cmp")

        scored_once = comparison.score.rank_divergence()
        scored_twice = scored_once.score.jensen_shannon_divergence()

        assert scored_twice._state.method == "jensen_shannon_divergence"
        assert scored_twice._state.is_weighted_avg is True
        assert scored_twice._state.parameters != scored_once._state.parameters

        # The second plan should produce weighted-avg columns, not simple score columns
        df_result = scored_twice._state.resolve()(df_mock, "word", "__wl_normed")
        assert "freq_diff" in df_result.columns
        assert "weighted_score_diff" in df_result.columns


class TestLexiconScorer:
    def test_lexicon_object_execution(self, df_mock) -> None:
        """A Lexicon object flows through Scorer.lexicon() to plan execution without error."""
        lexicon = Lexicon(
            lf=pl.LazyFrame({"token": ["the", "cat", "dog"], "score": [1.0, 5.0, -3.0]}),
            center=5.0,
        )
        comparison = Comparison("ref", "cmp")
        df_result = comparison.score.lexicon(lexicon=lexicon)._state.resolve()(
            df_mock, "word", "__wl_normed"
        )

        assert SCORE_COL in df_result.columns
        assert df_result.height == df_mock.height

    def test_default_reference_score_is_lexicon_center(self) -> None:
        """When no reference_score is given, it defaults to lexicon.center."""
        lexicon = Lexicon(
            lf=pl.LazyFrame({"token": ["the", "cat", "dog"], "score": [1.0, 5.0, -3.0]}),
            center=5.0,
        )
        comparison = Comparison("ref", "cmp")
        result = comparison.score.lexicon(lexicon=lexicon)

        assert result._state.parameters["reference_score"] == lexicon.center

    def test_explicit_center_resolves_to_lexicon_center(self) -> None:
        """reference_score='center' resolves to lexicon.center."""
        lexicon = Lexicon(
            lf=pl.LazyFrame({"token": ["the", "cat", "dog"], "score": [1.0, 5.0, -3.0]}),
            center=5.0,
        )
        comparison = Comparison("ref", "cmp")
        result = comparison.score.lexicon(lexicon=lexicon, reference_score="center")

        assert result._state.parameters["reference_score"] == lexicon.center

    def test_center_without_lexicon_raises(self) -> None:
        """reference_score='center' raises ValueError when no Lexicon object is provided."""
        comparison = Comparison("ref", "cmp")

        with pytest.raises(ValueError, match="center"):
            comparison.score.lexicon(lexicon={"the": 1.0}, reference_score="center")

    def test_weighted_avg_output_columns(self, df_mock) -> None:
        """Plan is wired to weighted_average_score_plan, producing all weighted-avg columns."""
        comparison = Comparison("ref", "cmp")
        df_result = comparison.score.lexicon(
            lexicon={"the": 1.0, "cat": 5.0, "dog": -3.0}
        )._state.resolve()(df_mock, "word", "__wl_normed")

        assert "freq_diff" in df_result.columns
        assert "weighted_score_diff" in df_result.columns
        assert "ref_score_diff" in df_result.columns
        assert SCORE_COL in df_result.columns

    def test_lexicon_scores_affect_comparison_score(self, df_mock) -> None:
        """Different lexicons produce different comparison scores, proving lexicon scores flow through."""
        comparison = Comparison("ref", "cmp")
        df_flat = comparison.score.lexicon(
            lexicon={"the": 1.0, "cat": 1.0, "dog": 1.0}
        )._state.resolve()(df_mock, "word", "__wl_normed")
        df_varied = comparison.score.lexicon(
            lexicon={"the": 1.0, "cat": 5.0, "dog": -3.0}
        )._state.resolve()(df_mock, "word", "__wl_normed")

        assert_frame_not_equal(
            df_flat.select(SCORE_COL),
            df_varied.select(SCORE_COL),
        )

    def test_lexicon_reference_comparison_routing(self, df_mock) -> None:
        """lexicon_reference and lexicon_comparison are wired to their respective sides."""
        lex_a = {"the": 1.0, "cat": 5.0, "dog": -3.0}
        lex_b = {"the": -1.0, "cat": 2.0, "dog": 8.0}

        comparison = Comparison("ref", "cmp")
        df_single = comparison.score.lexicon(lexicon=lex_a)._state.resolve()(
            df_mock, "word", "__wl_normed"
        )
        df_split = comparison.score.lexicon(
            lexicon_reference=lex_a, lexicon_comparison=lex_b
        )._state.resolve()(df_mock, "word", "__wl_normed")

        assert_frame_not_equal(
            df_single.select(SCORE_COL),
            df_split.select(SCORE_COL),
        )

    def test_impute_parameter_wiring(self, df_mock) -> None:
        """Different `impute` strategies produce different outputs when the lexicon is partial."""
        lexicon = {"the": 1.0, "cat": 5.0}  # missing "dog"
        comparison = Comparison("ref", "cmp")

        df_borrow = comparison.score.lexicon(lexicon=lexicon, impute="borrow")._state.resolve()(
            df_mock, "word", "__wl_normed"
        )
        df_zero = comparison.score.lexicon(lexicon=lexicon, impute=0.0)._state.resolve()(
            df_mock, "word", "__wl_normed"
        )

        # With single-lexicon borrow + drop_null_scores=True, "dog" is dropped.
        # With impute=0.0, "dog" is kept with score 0. Heights differ.
        assert df_borrow.height != df_zero.height

    def test_borrowed_flag_survives_scoring(self, df_mock) -> None:
        """`is_borrowed` is passed through the weighted-average plan into the scored dataframe."""
        comparison = Comparison("ref", "cmp")
        df_result = comparison.score.lexicon(
            lexicon_reference={"the": 1.0, "cat": 5.0},  # missing "dog"
            lexicon_comparison={"the": 2.0, "dog": -3.0},  # missing "cat"
            impute="borrow",
        )._state.resolve()(df_mock, "word", "__wl_normed")

        borrowed = df_result.filter(col(IS_BORROWED_COL))["word"].to_list()
        assert set(borrowed) == {"cat", "dog"}

    def test_no_borrowed_flag_for_other_impute_strategies(self, df_mock) -> None:
        """The `is_borrowed` column is only added when imputing by borrowing."""
        comparison = Comparison("ref", "cmp")
        df_result = comparison.score.lexicon(
            lexicon={"the": 1.0, "cat": 5.0}, impute=0.0
        )._state.resolve()(df_mock, "word", "__wl_normed")

        assert IS_BORROWED_COL not in df_result.columns

    def test_tokens_without_lexicon_scores_are_dropped(self, df_mock) -> None:
        """
        A token the lexicon does not score has no score to weight, so it cannot contribute to the
        weighted average and is dropped from the comparison.
        """
        comparison = Comparison("ref", "cmp")
        df_result = comparison.score.lexicon(lexicon={"the": 1.0, "cat": 5.0})._state.resolve()(
            df_mock, "word", "__wl_normed"
        )

        assert df_result["word"].to_list() == ["the", "cat"]
