import polars as pl
import pytest
from polars import col
from polars.testing import assert_frame_equal, assert_frame_not_equal

from wordlevel.api.comparison.comparison import Comparison
from wordlevel.api.comparison.score import Scorer
from wordlevel.constants import (
    COMPARISON_ATTRIBUTION,
    IS_BORROWED_COL,
    LEXICON_SCORE_REF_COL,
    NORMED_FREQ_REF_COL,
    REFERENCE_ATTRIBUTION,
    REINFORCE_ATTRIBUTION,
    SCORE_COL,
)
from wordlevel.lexicons.core.load import Lexicon


@pytest.fixture
def df_mock() -> pl.DataFrame:
    """Create a mock DataFrame for testing plan execution."""
    return pl.DataFrame(
        {
            "word": ["the", "cat", "dog"],
            "ref": [100, 50, 25],
            "cmp": [20, 30, 80],
        }
    )


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
                ["reference_score", "scored_in_one_lexicon"],
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
        df_result = comparison.score.frequency()._state.resolve()(df_mock, "word")

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
        df_result = result._state.resolve()(df_mock, "word")

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
        df_result = score_func(comparison.score)._state.resolve()(df_mock, "word")

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
        df_result = score_func(comparison.score)._state.resolve()(df_mock, "word")

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
        )
        comparison = Comparison("ref", "cmp")
        plan = comparison.score.kullback_leibler_divergence()._state.resolve()

        with pytest.raises(ValueError, match="Kullback-Leibler divergence is infinite"):
            plan(df, "word")

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

        df_result1 = result1._state.resolve()(df_mock, "word")
        df_result2 = result2._state.resolve()(df_mock, "word")

        assert_frame_not_equal(
            df_result1, df_result2, check_row_order=False, check_column_order=False
        )

    @pytest.mark.parametrize(
        "score_func",
        [
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
        result_ref3 = score_func(comparison.score, col(NORMED_FREQ_REF_COL).pow(2).sum())

        df_ref1 = result_ref1._state.resolve()(df_mock, "word")
        df_ref2 = result_ref2._state.resolve()(df_mock, "word")
        df_ref3 = result_ref3._state.resolve()(df_mock, "word")

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
        df_method = method_result._state.resolve()(df_mock, "word")
        df_alias = alias_result._state.resolve()(df_mock, "word")

        assert_frame_equal(df_method, df_alias, check_row_order=False, check_column_order=False)


class TestEdgeCases:
    @pytest.mark.parametrize(
        "score_func",
        [
            lambda s: s.rank_divergence(),
            lambda s: s.frequency(),
            lambda s: s.proportion(),
            lambda s: s.tsallis_entropy(),
            lambda s: s.kullback_leibler_divergence(),
            lambda s: s.jensen_shannon_divergence(),
            lambda s: s.lexicon(lexicon={"the": 1.0, "cat": 5.0, "dog": -3.0}),
        ],
    )
    def test_scoring_only_needs_the_raw_freq_columns(self, score_func):
        """Every plan normalizes the frequencies it needs, so raw corpus columns are enough."""
        df_raw = pl.DataFrame(
            {
                "word": ["the", "cat", "dog"],
                "my_ref": [100, 50, 25],
                "my_cmp": [80, 60, 20],
            }
        )

        comparison = Comparison("my_ref", "my_cmp")
        result = score_func(comparison.score)

        assert result._state.resolve()(df_raw, "word").lazy().collect().height > 0

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
    def test_plan_uses_provided_token_col(self, df_mock, score_func):
        """Plans use the token_col passed at execution time, not a hardcoded value."""
        comparison = Comparison("ref", "cmp")
        result = score_func(comparison.score)

        df_renamed = df_mock.rename({"word": "token"})

        df_result_default = result._state.resolve()(df_mock, "word")
        df_result_renamed = result._state.resolve()(df_renamed, "token")

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
            result._state.resolve()(df_mock, "word")

    def test_calling_score_twice_overrides_previous(self, df_mock):
        """Calling a second scoring method on a scored Comparison overrides the first."""
        comparison = Comparison("ref", "cmp")

        scored_once = comparison.score.rank_divergence()
        scored_twice = scored_once.score.jensen_shannon_divergence()

        assert scored_twice._state.method == "jensen_shannon_divergence"
        assert scored_twice._state.is_weighted_avg is True
        assert scored_twice._state.parameters != scored_once._state.parameters

        # The second plan should produce weighted-avg columns, not simple score columns
        df_result = scored_twice._state.resolve()(df_mock, "word")
        assert "freq_diff" in df_result.columns
        assert "weighted_score_diff" in df_result.columns


class TestLexiconScorer:
    def test_lexicon_join_is_its_own_plan(self, df_mock) -> None:
        """
        The join runs as a plan of its own rather than nested inside the score plan, so that
        operations can be slotted between joining a lexicon and scoring with it.
        """
        comparison = Comparison("ref", "cmp").score.lexicon(lexicon={"the": 1.0, "cat": 5.0})
        join_plan = comparison._state.lexicon_join_plan

        assert join_plan is not None

        df_joined = join_plan(df_mock, "word")

        assert LEXICON_SCORE_REF_COL in df_joined.columns
        assert SCORE_COL not in df_joined.columns

    def test_rescoring_without_a_lexicon_clears_the_join_plan(self) -> None:
        """A lexicon join belongs to the lexicon method, so scoring another way discards it."""
        comparison = Comparison("ref", "cmp").score.lexicon(lexicon={"the": 1.0, "cat": 5.0})

        assert comparison.score.shannon_entropy()._state.lexicon_join_plan is None

    def test_lexicon_object_execution(self, df_mock) -> None:
        """A Lexicon object flows through Scorer.lexicon() to plan execution without error."""
        lexicon = Lexicon(
            lf=pl.LazyFrame({"token": ["the", "cat", "dog"], "score": [1.0, 5.0, -3.0]}),
            center=5.0,
        )
        comparison = Comparison("ref", "cmp")
        df_result = comparison.score.lexicon(lexicon=lexicon)._state.resolve()(df_mock, "word")

        assert SCORE_COL in df_result.columns
        assert df_result.height == df_mock.height

    def test_default_reference_score_for_lexicons_is_zero(self) -> None:
        """When no reference_score is given, it defaults to lexicon.center."""
        lexicon = Lexicon(
            lf=pl.LazyFrame({"token": ["the", "cat", "dog"], "score": [1.0, 5.0, -3.0]}),
            center=5.0,
        )
        comparison = Comparison("ref", "cmp")
        result = comparison.score.lexicon(lexicon=lexicon)

        assert result._state.parameters["reference_score"] == 0

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
        )._state.resolve()(df_mock, "word")

        assert "freq_diff" in df_result.columns
        assert "weighted_score_diff" in df_result.columns
        assert "ref_score_diff" in df_result.columns
        assert SCORE_COL in df_result.columns

    def test_lexicon_scores_affect_comparison_score(self, df_mock) -> None:
        """Different lexicons produce different comparison scores, proving lexicon scores flow through."""
        comparison = Comparison("ref", "cmp")
        df_flat = comparison.score.lexicon(
            lexicon={"the": 1.0, "cat": 1.0, "dog": 1.0}
        )._state.resolve()(df_mock, "word")
        df_varied = comparison.score.lexicon(
            lexicon={"the": 1.0, "cat": 5.0, "dog": -3.0}
        )._state.resolve()(df_mock, "word")

        assert_frame_not_equal(
            df_flat.select(SCORE_COL),
            df_varied.select(SCORE_COL),
        )

    def test_lexicon_reference_comparison_routing(self, df_mock) -> None:
        """lexicon_reference and lexicon_comparison are wired to their respective sides."""
        lex_a = {"the": 1.0, "cat": 5.0, "dog": -3.0}
        lex_b = {"the": -1.0, "cat": 2.0, "dog": 8.0}

        comparison = Comparison("ref", "cmp")
        df_single = comparison.score.lexicon(lexicon=lex_a)._state.resolve()(df_mock, "word")
        df_split = comparison.score.lexicon(
            lexicon_reference=lex_a, lexicon_comparison=lex_b
        )._state.resolve()(df_mock, "word")

        assert_frame_not_equal(
            df_single.select(SCORE_COL),
            df_split.select(SCORE_COL),
        )

    def test_scored_in_one_lexicon_parameter_wiring(self, df_mock) -> None:
        """Borrowing and imputing score a token that only one lexicon scores differently."""
        df_borrow, df_zero = [
            Comparison("ref", "cmp")
            .score.lexicon(
                lexicon_reference={"the": 1.0, "cat": 5.0},  # missing "dog"
                lexicon_comparison={"the": 2.0, "cat": 5.0, "dog": -3.0},
                scored_in_one_lexicon=scored_in_one_lexicon,
            )
            ._state.resolve()(df_mock, "word")
            for scored_in_one_lexicon in ("borrow", 0.0)
        ]

        assert_frame_not_equal(df_borrow.select(SCORE_COL), df_zero.select(SCORE_COL))

    def test_invalid_scored_in_one_lexicon_raises(self) -> None:
        """An unrecognized string is rejected when the comparison is built, not when resolved."""
        with pytest.raises(ValueError, match="scored_in_one_lexicon"):
            Comparison("ref", "cmp").score.lexicon(
                lexicon={"the": 1.0},
                scored_in_one_lexicon="median",  # type: ignore[arg-type]
            )

    def test_borrowed_flag_survives_scoring(self, df_mock) -> None:
        """`is_borrowed` is passed through the weighted-average plan into the scored dataframe."""
        comparison = Comparison("ref", "cmp")
        df_result = comparison.score.lexicon(
            lexicon_reference={"the": 1.0, "cat": 5.0},  # missing "dog"
            lexicon_comparison={"the": 2.0, "dog": -3.0},  # missing "cat"
            scored_in_one_lexicon="borrow",
        )._state.resolve()(df_mock, "word")

        borrowed = df_result.filter(col(IS_BORROWED_COL))["word"].to_list()
        assert set(borrowed) == {"cat", "dog"}

    def test_no_borrowed_flag_unless_borrowing(self, df_mock) -> None:
        """The `is_borrowed` column is only added when borrowing."""
        comparison = Comparison("ref", "cmp")
        df_result = comparison.score.lexicon(
            lexicon={"the": 1.0, "cat": 5.0}, scored_in_one_lexicon=0.0
        )._state.resolve()(df_mock, "word")

        assert IS_BORROWED_COL not in df_result.columns

    def test_tokens_without_lexicon_scores_are_dropped(self, df_mock) -> None:
        """
        A token the lexicon does not score has no score to weight, so it cannot contribute to the
        weighted average and is dropped from the comparison.
        """
        comparison = Comparison("ref", "cmp")
        df_result = comparison.score.lexicon(lexicon={"the": 1.0, "cat": 5.0})._state.resolve()(
            df_mock, "word"
        )

        assert df_result["word"].to_list() == ["the", "cat"]

    def test_imputing_drops_tokens_no_lexicon_scores(self, df_mock) -> None:
        """
        Imputation only fills a token that one of two lexicons scores. A token the lexicon does not
        score at all is still dropped, rather than given the imputed value.
        """
        df_result = (
            Comparison("ref", "cmp")
            .score.lexicon(lexicon={"the": 1.0, "cat": 5.0}, scored_in_one_lexicon=0.0)
            ._state.resolve()(df_mock, "word")
        )

        assert df_result["word"].to_list() == ["the", "cat"]

    def test_imputing_changes_the_normalized_freqs(self, df_mock) -> None:
        """
        Imputing keeps a token that only one lexicon scores, so that token holds onto its share of
        the frequency mass instead of the mass being spread over the tokens both lexicons score.
        """
        dropped, imputed = [
            Comparison("ref", "cmp")
            .score.lexicon(
                lexicon_reference={"the": 1.0, "cat": 5.0},  # missing "dog"
                lexicon_comparison={"the": 2.0, "cat": 5.0, "dog": -3.0},
                scored_in_one_lexicon=scored_in_one_lexicon,
            )
            ._state.resolve()(df_mock, "word")
            .filter(col("word") == "the")["ref__wl_normed"]
            .item()
            for scored_in_one_lexicon in ("drop", 0.0)
        ]

        assert dropped > imputed

    def test_mean_reference_score_is_taken_over_the_scored_vocabulary(self, df_mock) -> None:
        """
        A `"mean"` reference score is the reference corpus's own weighted average, and it is taken
        over the tokens the lexicon scores rather than being shrunk by the unscored mass.
        """
        lexicon = {"the": 1.0, "cat": 5.0}  # "dog" is in the corpora but not the lexicon
        df_result = (
            Comparison("ref", "cmp")
            .score.lexicon(lexicon=lexicon, reference_score="mean")
            ._state.resolve()(df_mock, "word")
        )

        # One lexicon scores both corpora alike, so a token's `ref_score_diff` is its own lexicon
        # score minus the reference score
        the = df_result.filter(col("word") == "the")
        reference_score = lexicon["the"] - the["ref_score_diff"].item()

        # The reference corpus is 100 "the" and 50 "cat" once "dog" is dropped
        assert reference_score == pytest.approx((100 * 1.0 + 50 * 5.0) / 150)

    def test_contributions_sum_to_the_difference_in_weighted_averages(self, df_mock) -> None:
        """
        The frequencies are renormalized after the lexicon drops the tokens it cannot score, so
        each corpus's weighted average is taken over the vocabulary that is actually scored.
        """
        lexicon = {"the": 1.0, "cat": 5.0}  # "dog" is in the corpora but not the lexicon
        df_result = (
            Comparison("ref", "cmp")
            .score.lexicon(lexicon=lexicon, reference_score=0)
            ._state.resolve()(df_mock, "word")
        )

        # The reference corpus is 100 "the" and 50 "cat"; the comparison is 20 and 30
        expected = (20 * 1.0 + 30 * 5.0) / 50 - (100 * 1.0 + 50 * 5.0) / 150

        assert df_result[SCORE_COL].sum() == pytest.approx(expected)

    def test_reference_score_does_not_change_the_total(self, df_mock) -> None:
        """
        A reference score shifts every token's contribution by its frequency difference, and those
        differences cancel once each corpus's frequencies sum to one over the scored vocabulary.
        """
        totals = [
            Comparison("ref", "cmp")
            .score.lexicon(lexicon={"the": 1.0, "cat": 5.0}, reference_score=reference_score)
            ._state.resolve()(df_mock, "word")[SCORE_COL]
            .sum()
            for reference_score in (0, "mean")
        ]

        assert totals[0] == pytest.approx(totals[1])
