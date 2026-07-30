from typing import Callable

import polars as pl
import pytest
from polars import col
from polars.testing import assert_frame_equal

from wordlevel.api.catalog.catalog import Catalog
from wordlevel.api.comparison.comparison import Comparison, comp
from wordlevel.api.comparison.score import Scorer


LEXICON = {"the": 1.0, "cat": 5.0, "dog": 2.0, "ran": 3.0}


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
