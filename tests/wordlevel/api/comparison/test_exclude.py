from collections.abc import Callable

import polars as pl
import pytest
from polars import col
from polars.testing import assert_frame_equal

from wordlevel.api.catalog.catalog import Catalog
from wordlevel.api.comparison.comparison import Comparison, comp
from wordlevel.api.comparison.score import Scorer
from wordlevel.core.comparison.exclude import ExcludeCorpus


LEXICON = {"the": 1.0, "cat": 5.0, "dog": 2.0, "ran": 3.0}

BY_RANGE_METHODS = ["by_frequency", "by_proportion", "by_lexicon_score"]
"""The exclusions that take a range, and so share one validation of it."""


@pytest.fixture
def df_mock() -> pl.DataFrame:
    """A mock corpus frame where every token appears in both corpora."""
    return pl.DataFrame(
        {
            "word": ["the", "cat", "dog", "ran"],
            "ref": [100, 50, 25, 10],
            "cmp": [80, 30, 60, 20],
        }
    )


class TestExcludeTokens:
    @pytest.mark.parametrize(
        "score_func",
        [
            lambda s: s.frequency(),
            lambda s: s.proportion(),
            lambda s: s.shannon_entropy(),
            lambda s: s.tsallis_entropy(alpha=2),
            lambda s: s.lexicon(lexicon=LEXICON),
            lambda s: s.kullback_leibler_divergence(),
            lambda s: s.jensen_shannon_divergence(),
            lambda s: s.rank_divergence(),
        ],
        ids=[
            "frequency",
            "proportion",
            "shannon_entropy",
            "tsallis_entropy",
            "lexicon",
            "kullback_leibler_divergence",
            "jensen_shannon_divergence",
            "rank_divergence",
        ],
    )
    def test_matches_a_catalog_built_without_the_tokens(
        self, df_mock: pl.DataFrame, score_func: Callable[[Scorer], Comparison]
    ) -> None:
        """
        Excluding a token is equivalent to never having it in the corpora, for every scoring
        method. Anything a measure derives from the vocabulary — normalized frequencies, ranks, the
        rank divergence's normalization constant — is derived from the vocabulary that remains.
        """
        excluded = ["the", "ran"]

        from_full = Catalog(df_mock, token_col="word", corpora=["ref", "cmp"]).with_comparisons(
            score_func(comp("ref", "cmp").exclude.tokens(excluded).score).alias("x")
        )
        from_filtered = Catalog(
            df_mock.filter(~col("word").is_in(excluded)), token_col="word", corpora=["ref", "cmp"]
        ).with_comparisons(score_func(comp("ref", "cmp").score).alias("x"))

        assert_frame_equal(
            from_full.comparisons["x"].df,
            from_filtered.comparisons["x"].df,
        )

    def test_excluding_a_token_outside_the_vocabulary_changes_nothing(
        self, df_mock: pl.DataFrame
    ) -> None:
        """Exclusion is a set operation on the vocabulary, so an absent token is not an error."""
        scored = comp("ref", "cmp").score.frequency()

        assert_frame_equal(
            scored.exclude.tokens(["absent"])._state.resolve()(df_mock, "word"),
            scored._state.resolve()(df_mock, "word"),
        )

    def test_excluding_more_than_once_removes_the_union(self, df_mock: pl.DataFrame) -> None:
        """Each call composes onto the exclusions already in place rather than replacing them."""
        df_result = (
            comp("ref", "cmp")
            .exclude.tokens(["the"])
            .exclude.tokens(["cat"])
            .score.frequency()
            ._state.resolve()(df_mock, "word")
        )

        assert df_result["word"].to_list() == ["dog", "ran"]

    def test_excluding_before_scoring_matches_excluding_after(self, df_mock: pl.DataFrame) -> None:
        """
        Exclusion occupies its own plan, so the order it is called in does not matter. Scoring in
        particular must not discard exclusions already in place, since they are the user's and not
        the scoring method's.
        """
        before = comp("ref", "cmp").exclude.tokens(["the"]).score.proportion()
        after = comp("ref", "cmp").score.proportion().exclude.tokens(["the"])

        df_before = before._state.resolve()(df_mock, "word")

        assert "the" not in df_before["word"].to_list()
        assert_frame_equal(df_before, after._state.resolve()(df_mock, "word"))


class TestExcludeUnsharedTokens:
    def test_matches_excluding_the_tokens_a_corpus_does_not_use(self) -> None:
        """`dog` is never used in the reference corpus and `ran` never in the comparison."""
        df_unshared = pl.DataFrame(
            {
                "word": ["the", "cat", "dog", "ran"],
                "ref": [100, 50, 0, 10],
                "cmp": [80, 30, 60, 0],
            }
        )

        unshared = comp("ref", "cmp").exclude.unshared_tokens().score.proportion()
        by_name = comp("ref", "cmp").exclude.tokens(["dog", "ran"]).score.proportion()

        assert_frame_equal(
            unshared._state.resolve()(df_unshared, "word"),
            by_name._state.resolve()(df_unshared, "word"),
        )

    def test_corpora_sharing_every_token_are_left_alone(self, df_mock: pl.DataFrame) -> None:
        """The exclusion is about the vocabulary the corpora share, not about rare tokens."""
        scored = comp("ref", "cmp").score.proportion()

        assert_frame_equal(
            scored.exclude.unshared_tokens()._state.resolve()(df_mock, "word"),
            scored._state.resolve()(df_mock, "word"),
        )


class TestExcludeByFrequency:
    def test_matches_excluding_the_same_tokens_directly(self, df_mock: pl.DataFrame) -> None:
        """
        A range of frequencies names a set of tokens, so excluding by frequency is equivalent to
        excluding those tokens by name. `the` is the only token above 70 in both corpora.
        """
        by_frequency = comp("ref", "cmp").exclude.by_frequency(greater_than=70)
        by_tokens = comp("ref", "cmp").exclude.tokens(["the"])

        assert_frame_equal(
            by_frequency.score.proportion()._state.resolve()(df_mock, "word"),
            by_tokens.score.proportion()._state.resolve()(df_mock, "word"),
        )

    def test_less_than_excludes_the_rare_end(self, df_mock: pl.DataFrame) -> None:
        """The bound is one-sided, so nothing above it is touched."""
        df_result = (
            comp("ref", "cmp")
            .exclude.by_frequency(less_than=25)
            .score.frequency()
            ._state.resolve()(df_mock, "word")
        )

        assert df_result["word"].to_list() == ["the", "cat", "dog"]

    def test_greater_than_excludes_the_common_end(self, df_mock: pl.DataFrame) -> None:
        """The bound is one-sided, so nothing below it is touched."""
        df_result = (
            comp("ref", "cmp")
            .exclude.by_frequency(greater_than=30)
            .score.frequency()
            ._state.resolve()(df_mock, "word")
        )

        assert df_result["word"].to_list() == ["dog", "ran"]

    @pytest.mark.parametrize(
        "inclusive, expected",
        [(True, ["the", "cat", "dog"]), (False, ["the", "cat", "dog", "ran"])],
    )
    def test_bounds_are_part_of_the_range_only_when_inclusive(
        self, df_mock: pl.DataFrame, inclusive: bool, expected: list[str]
    ) -> None:
        """`ran` appears 20 times in the comparison corpus, so it sits exactly on the bound."""
        df_result = (
            comp("ref", "cmp")
            .exclude.by_frequency(less_than=20, inclusive=inclusive)
            .score.frequency()
            ._state.resolve()(df_mock, "word")
        )

        assert df_result["word"].to_list() == expected

    def test_excluding_more_than_once_removes_the_union(self, df_mock: pl.DataFrame) -> None:
        """Each call composes onto the exclusions already in place, so the ranges accumulate."""
        df_result = (
            comp("ref", "cmp")
            .exclude.by_frequency(less_than=20)
            .exclude.by_frequency(greater_than=70)
            .score.frequency()
            ._state.resolve()(df_mock, "word")
        )

        assert df_result["word"].to_list() == ["cat", "dog"]

    @pytest.mark.parametrize(
        "corpus, expected",
        [
            ("reference", ["the", "cat"]),
            ("comparison", ["the", "dog"]),
            ("both", ["the", "cat", "dog"]),
            ("either", ["the"]),
        ],
    )
    def test_corpus_chooses_which_frequency_is_tested(
        self, df_mock: pl.DataFrame, corpus: ExcludeCorpus, expected: list[str]
    ) -> None:
        """
        The two corpora give a token different frequencies, so which of them the range is tested
        against decides what is excluded. `dog` is rare only in the reference and `cat` only in the
        comparison, while `ran` is rare in both.
        """
        df_result = (
            comp("ref", "cmp")
            .exclude.by_frequency(less_than=30, corpus=corpus)
            .score.frequency()
            ._state.resolve()(df_mock, "word")
        )

        assert df_result["word"].to_list() == expected


class TestExcludeByProportion:
    def test_matches_excluding_the_same_tokens_directly(self, df_mock: pl.DataFrame) -> None:
        """
        A range of proportions names a set of tokens, so excluding by proportion is equivalent to
        excluding those tokens by name. `ran` is the only token under 15% of both corpora.
        """
        by_proportion = comp("ref", "cmp").exclude.by_proportion(less_than=0.15)
        by_tokens = comp("ref", "cmp").exclude.tokens(["ran"])

        assert_frame_equal(
            by_proportion.score.proportion()._state.resolve()(df_mock, "word"),
            by_tokens.score.proportion()._state.resolve()(df_mock, "word"),
        )

    def test_proportions_are_taken_over_the_corpus_as_given(self, df_mock: pl.DataFrame) -> None:
        """
        Proportions are normalized before anything narrows the vocabulary, so a token excluded by
        name does not change the shares the range is tested against. Were they taken over what
        survives, dropping `the` would lift `ran` to 18% of the comparison corpus and keep it.
        """
        excluded_first = (
            comp("ref", "cmp").exclude.tokens(["the"]).exclude.by_proportion(less_than=0.15)
        )
        excluded_last = (
            comp("ref", "cmp").exclude.by_proportion(less_than=0.15).exclude.tokens(["the"])
        )

        df_result = excluded_first.score.proportion()._state.resolve()(df_mock, "word")

        assert df_result["word"].to_list() == ["cat", "dog"]
        assert_frame_equal(
            df_result, excluded_last.score.proportion()._state.resolve()(df_mock, "word")
        )

    def test_proportions_do_not_depend_on_the_lexicons_coverage(
        self, df_mock: pl.DataFrame
    ) -> None:
        """
        The lexicon join drops the tokens it cannot score, so normalizing after it would make the
        range depend on how the comparison is scored. `the` is dropped for want of a score, and
        `ran` is still under 15% of the corpora the catalog gave.
        """
        df_result = (
            comp("ref", "cmp")
            .exclude.by_proportion(less_than=0.15)
            .score.lexicon(lexicon={"cat": 5.0, "dog": 2.0, "ran": 3.0})
            ._state.resolve()(df_mock, "word")
        )

        assert df_result["word"].to_list() == ["cat", "dog"]


class TestExcludeByRangeValidation:
    @pytest.mark.parametrize("method", BY_RANGE_METHODS)
    def test_no_range_raises(self, method: str) -> None:
        """
        Without a bound there is no range to exclude. Plans only run once a catalog resolves the
        comparison, so validating in the api layer is what makes this an error at the call that
        made it rather than much later.
        """
        exclude_by_range = getattr(comp("ref", "cmp").exclude, method)

        with pytest.raises(ValueError, match="Must specify one of"):
            exclude_by_range()

    @pytest.mark.parametrize("method", BY_RANGE_METHODS)
    def test_more_than_one_range_raises(self, method: str) -> None:
        """Two bounds in one call are ambiguous between excluding their union and intersection."""
        exclude_by_range = getattr(comp("ref", "cmp").exclude, method)

        with pytest.raises(ValueError, match="mutually exclusive"):
            exclude_by_range(less_than=0.2, greater_than=0.4)

    @pytest.mark.parametrize("method", BY_RANGE_METHODS)
    def test_reversed_between_raises(self, method: str) -> None:
        """A range whose bounds are the wrong way around excludes nothing, silently."""
        exclude_by_range = getattr(comp("ref", "cmp").exclude, method)

        with pytest.raises(ValueError, match="lower bound"):
            exclude_by_range(between=(0.4, 0.2))

    @pytest.mark.parametrize("method", BY_RANGE_METHODS)
    def test_unrecognized_corpus_raises(self, method: str) -> None:
        """The range has to be tested against one of the corpora being compared."""
        exclude_by_range = getattr(comp("ref", "cmp").exclude, method)

        with pytest.raises(ValueError, match="`corpus` must be one of"):
            exclude_by_range(less_than=0.2, corpus="neither")

    @pytest.mark.parametrize(
        "bounds",
        [{"less_than": 5}, {"greater_than": 1.5}, {"between": (0.5, 2.0)}],
        ids=["less_than", "greater_than", "between"],
    )
    def test_proportions_outside_zero_to_one_raise(
        self, bounds: dict[str, float | tuple[float, float]]
    ) -> None:
        """A bound above 1 is a count, which would quietly exclude all or none of the vocabulary."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            comp("ref", "cmp").exclude.by_proportion(**bounds)  # type: ignore[arg-type]


class TestExcludeByLexiconScore:
    def test_matches_excluding_the_same_tokens_directly(self, df_mock: pl.DataFrame) -> None:
        """
        A range of lexicon scores names a set of tokens, so excluding by score is equivalent to
        excluding those tokens by name. `dog` scores 2.0 and `ran` scores 3.0.
        """
        by_score = comp("ref", "cmp").exclude.by_lexicon_score(between=(2.0, 3.0))
        by_tokens = comp("ref", "cmp").exclude.tokens(["dog", "ran"])

        assert_frame_equal(
            by_score.score.lexicon(lexicon=LEXICON)._state.resolve()(df_mock, "word"),
            by_tokens.score.lexicon(lexicon=LEXICON)._state.resolve()(df_mock, "word"),
        )

    def test_less_than_excludes_the_low_end(self, df_mock: pl.DataFrame) -> None:
        """The bound is one-sided, so nothing above it is touched."""
        df_result = (
            comp("ref", "cmp")
            .exclude.by_lexicon_score(less_than=2.0)
            .score.lexicon(lexicon=LEXICON)
            ._state.resolve()(df_mock, "word")
        )

        assert df_result["word"].to_list() == ["cat", "ran"]

    def test_greater_than_excludes_the_high_end(self, df_mock: pl.DataFrame) -> None:
        """The bound is one-sided, so nothing below it is touched."""
        df_result = (
            comp("ref", "cmp")
            .exclude.by_lexicon_score(greater_than=3.0)
            .score.lexicon(lexicon=LEXICON)
            ._state.resolve()(df_mock, "word")
        )

        assert df_result["word"].to_list() == ["the", "dog"]

    @pytest.mark.parametrize(
        "inclusive, expected", [(True, ["cat"]), (False, ["the", "cat", "ran"])]
    )
    def test_bounds_are_part_of_the_range_only_when_inclusive(
        self, df_mock: pl.DataFrame, inclusive: bool, expected: list[str]
    ) -> None:
        """`the` scores exactly 1.0 and `ran` exactly 3.0, so both sit on a bound of the range."""
        df_result = (
            comp("ref", "cmp")
            .exclude.by_lexicon_score(between=(1.0, 3.0), inclusive=inclusive)
            .score.lexicon(lexicon=LEXICON)
            ._state.resolve()(df_mock, "word")
        )

        assert df_result["word"].to_list() == expected

    def test_excluding_more_than_once_removes_the_union(self, df_mock: pl.DataFrame) -> None:
        """Each call composes onto the exclusions already in place, so the ranges accumulate."""
        df_result = (
            comp("ref", "cmp")
            .exclude.by_lexicon_score(less_than=1.0)
            .exclude.by_lexicon_score(greater_than=5.0)
            .score.lexicon(lexicon=LEXICON)
            ._state.resolve()(df_mock, "word")
        )

        assert df_result["word"].to_list() == ["dog", "ran"]

    @pytest.mark.parametrize(
        "corpus, expected",
        [
            ("reference", ["cat", "ran"]),
            ("comparison", ["cat", "dog"]),
            ("both", ["cat", "dog", "ran"]),
            ("either", ["cat"]),
        ],
    )
    def test_corpus_chooses_which_lexicons_score_is_tested(
        self, df_mock: pl.DataFrame, corpus: ExcludeCorpus, expected: list[str]
    ) -> None:
        """
        With a lexicon per corpus the two scores differ, so which of them the range is tested
        against decides what is excluded. `dog` is low-scoring only in the reference and `ran` only
        in the comparison, while `the` is low-scoring in both.
        """
        df_result = (
            comp("ref", "cmp")
            .exclude.by_lexicon_score(less_than=2.5, corpus=corpus)
            .score.lexicon(
                lexicon_reference={"the": 1.0, "cat": 5.0, "dog": 2.0, "ran": 3.0},
                lexicon_comparison={"the": 1.0, "cat": 5.0, "dog": 5.0, "ran": 1.0},
            )
            ._state.resolve()(df_mock, "word")
        )

        assert df_result["word"].to_list() == expected

    def test_excluding_before_scoring_matches_excluding_after(self, df_mock: pl.DataFrame) -> None:
        """
        The exclusion shares a plan with `exclude.tokens` but is only expressible once the lexicon
        is joined, so it must run after the join no matter which order the two are called in.
        """
        before = comp("ref", "cmp").exclude.by_lexicon_score(less_than=2.0)
        after = (
            comp("ref", "cmp")
            .score.lexicon(lexicon=LEXICON)
            .exclude.by_lexicon_score(less_than=2.0)
        )

        assert_frame_equal(
            before.score.lexicon(lexicon=LEXICON)._state.resolve()(df_mock, "word"),
            after._state.resolve()(df_mock, "word"),
        )


class TestExcludeByLexiconScoreValidation:
    def test_excluding_an_already_unlexiconed_comparison_raises(self) -> None:
        """A comparison scored another way has no lexicon scores to test a range against."""
        with pytest.raises(ValueError, match="scored by a lexicon"):
            comp("ref", "cmp").score.shannon_entropy().exclude.by_lexicon_score(less_than=2.0)

    def test_scoring_without_a_lexicon_after_excluding_raises(self) -> None:
        """The scoring method is only known to be incompatible once every plan is in place."""
        excluded = comp("ref", "cmp").exclude.by_lexicon_score(less_than=2.0)

        with pytest.raises(ValueError, match="scored by a lexicon"):
            excluded.score.shannon_entropy()._state.resolve()

    @pytest.mark.parametrize("impute", ["mean", 5.0])
    def test_imputed_scores_raise(self, impute: float | str) -> None:
        """
        An imputed score is a modeling choice rather than what the lexicon says about a token, so
        excluding by it would remove tokens on the strength of a value the lexicon never gave.
        """
        with pytest.raises(ValueError, match="modeling choice"):
            comp("ref", "cmp").score.lexicon(
                lexicon=LEXICON,
                impute=impute,  # type: ignore[arg-type]
            ).exclude.by_lexicon_score(less_than=2.0)

    def test_borrowed_scores_are_allowed(self, df_mock: pl.DataFrame) -> None:
        """A borrowed score is a genuine one, just taken from the lexicon of the other corpus."""
        df_result = (
            comp("ref", "cmp")
            .exclude.by_lexicon_score(less_than=2.0)
            .score.lexicon(lexicon=LEXICON, impute="borrow")
            ._state.resolve()(df_mock, "word")
        )

        assert df_result["word"].to_list() == ["cat", "ran"]


class TestExclusionsRecord:
    def test_default_exclusions_are_empty(self) -> None:
        """A comparison that excludes nothing has nothing to record."""
        assert comp("ref", "cmp")._state.exclusions == {}

    def test_tokens_are_recorded_as_their_union(self) -> None:
        """The record is keyed by the kind of exclusion, and repeated calls accumulate into it."""
        excluded_once = comp("ref", "cmp").exclude.tokens(["the"])
        excluded_twice = excluded_once.exclude.tokens(["cat", "the"])

        assert excluded_twice._state.exclusions == {"tokens": {"the", "cat"}}
        assert excluded_once._state.exclusions == {"tokens": {"the"}}

    def test_lexicon_score_ranges_are_recorded_in_the_order_they_were_applied(self) -> None:
        """Repeated calls exclude the union of their ranges, so every one of them is recorded."""
        excluded = (
            comp("ref", "cmp")
            .exclude.by_lexicon_score(less_than=2.0)
            .exclude.by_lexicon_score(between=(4.0, 5.0), inclusive=False, corpus="either")
        )

        assert excluded._state.exclusions == {
            "lexicon_score": [
                {"less_than": 2.0, "inclusive": True, "corpus": "both"},
                {"between": (4.0, 5.0), "inclusive": False, "corpus": "either"},
            ]
        }

    def test_unshared_tokens_are_recorded_as_having_been_excluded(self) -> None:
        """The exclusion takes no parameters, so there is nothing to record but that it ran."""
        excluded = comp("ref", "cmp").exclude.unshared_tokens()

        assert excluded._state.exclusions == {"unshared_tokens": True}

    def test_each_kind_of_range_is_recorded_under_its_own_key(self) -> None:
        """A comparison can exclude by more than one kind of range, and they do not collide."""
        excluded = (
            comp("ref", "cmp")
            .exclude.by_frequency(less_than=5)
            .exclude.by_proportion(greater_than=0.5, inclusive=False, corpus="either")
        )

        assert excluded._state.exclusions == {
            "frequency": [{"less_than": 5, "inclusive": True, "corpus": "both"}],
            "proportion": [{"greater_than": 0.5, "inclusive": False, "corpus": "either"}],
        }

    def test_result_reports_the_exclusions(self, df_mock: pl.DataFrame) -> None:
        """A result carries the record, so a scored comparison shows what was left out of it."""
        catalog = Catalog(df_mock, token_col="word", corpora=["ref", "cmp"]).with_comparisons(
            comp("ref", "cmp").exclude.tokens(["the"]).score.frequency().alias("x")
        )

        assert catalog.comparisons["x"].exclusions == {"tokens": {"the"}}
