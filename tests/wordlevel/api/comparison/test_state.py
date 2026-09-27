import polars as pl
import pytest
from polars.testing import assert_frame_equal

from wordlevel.api.comparison.comparison import comp
from wordlevel.api.comparison.state import ComparisonState
from wordlevel.constants import (
    FREQ_CMP_COL,
    FREQ_REF_COL,
    NORMED_FREQ_CMP_COL,
    NORMED_FREQ_REF_COL,
)


class TestComposePlans:
    def test_wrapping_plan_is_applied_to_the_base_plans_output(self) -> None:
        """Composing runs the base plan first and feeds its output to the wrapping plan."""
        composed = ComparisonState.compose_plans(
            lambda df, token_col: df.with_columns(pl.lit(1).alias("first")),
            lambda df, token_col: df.with_columns(pl.col("first").alias("second")),
        )

        assert composed(pl.DataFrame({"word": ["a"]}), "word").columns == [
            "word",
            "first",
            "second",
        ]

    def test_wrapping_plan_becomes_the_plan_when_there_is_no_base(self) -> None:
        """With nothing to compose onto, the wrapping plan is returned unchanged."""

        def plan(df: pl.DataFrame, token_col: str) -> pl.DataFrame:
            return df

        assert ComparisonState.compose_plans(None, plan) is plan


class TestResolve:
    def test_raises_without_a_score_plan(self) -> None:
        """A comparison that is only normalized has no scores to normalize."""
        with pytest.raises(ValueError, match="scored"):
            comp("ref", "cmp").normalize("total_diff")._state.resolve()

    def test_runs_the_comparisons_own_plans_around_the_users(self) -> None:
        """
        The comparison selects and renormalizes the two corpora before any plan of the user's
        scores them, and renames the columns back afterward.
        """
        seen: list[tuple[str, list[str]]] = []

        def record(name: str):
            def plan(
                df: pl.DataFrame | pl.LazyFrame, token_col: str
            ) -> pl.DataFrame | pl.LazyFrame:
                seen.append((name, df.lazy().collect_schema().names()))
                return df

            return plan

        state = ComparisonState(
            reference="ref",
            comparison="cmp",
            lexicon_join_plan=record("lexicon_join"),
            exclusion_plan=record("exclusion"),
            score_plan=record("score"),
        )
        df_result = state.resolve()(
            pl.DataFrame({"word": ["a"], "ref": [1], "cmp": [2], "other": [3]}), "word"
        )

        # The join runs on the two selected corpora, before they are renormalized and before any
        # other corpus in the catalog could reach them, so exclusions can see the lexicon scores
        assert seen[0] == ("lexicon_join", ["word", FREQ_REF_COL, FREQ_CMP_COL])
        assert seen[1] == ("exclusion", ["word", FREQ_REF_COL, FREQ_CMP_COL])

        name, columns = seen[2]
        assert name == "score"
        assert NORMED_FREQ_REF_COL in columns and NORMED_FREQ_CMP_COL in columns

        assert df_result.columns == ["word", "ref", "cmp", "ref__wl_normed", "cmp__wl_normed"]

    def test_drops_tokens_that_neither_corpus_uses(self) -> None:
        """
        A comparison's vocabulary is the one its two corpora use, not the catalog's, so a token
        another corpus contributed is dropped rather than carried as a row of zeros.
        """
        df_mock = pl.DataFrame({"word": ["a", "b"], "ref": [1, 0], "cmp": [2, 0], "other": [0, 5]})

        df_result = comp("ref", "cmp").score.frequency()._state.resolve()(df_mock, "word")

        assert df_result["word"].to_list() == ["a"]

    def test_normalizing_before_scoring_matches_normalizing_after(self) -> None:
        """Normalization occupies its own plan, so the order it is called in does not matter."""
        df_mock = pl.DataFrame(
            {"word": ["the", "cat", "dog"], "ref": [100, 50, 25], "cmp": [20, 30, 80]}
        )
        before = comp("ref", "cmp").normalize("total_diff").score.shannon_entropy()
        after = comp("ref", "cmp").score.shannon_entropy().normalize("total_diff")

        assert_frame_equal(
            before._state.resolve()(df_mock, "word").lazy().collect(),
            after._state.resolve()(df_mock, "word").lazy().collect(),
        )

    def test_invalid_normalization_for_a_later_scoring_method_raises(self) -> None:
        """The scoring method is only known to be compatible once every plan is in place."""
        normalized = comp("ref", "cmp").normalize("net_diff")

        with pytest.raises(ValueError, match="proportion"):
            normalized.score.proportion()._state.resolve()
