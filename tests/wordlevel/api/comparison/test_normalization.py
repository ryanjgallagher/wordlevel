from typing import Callable

import pytest
import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

from wordlevel.api.comparison.comparison import Comparison, comp
from wordlevel.api.comparison.score import Scorer
from wordlevel.constants import SCORE_COL, NORMALIZATION_CONSTANT_COL


@pytest.fixture
def df_mock() -> pl.DataFrame:
    """A mock corpus frame."""
    return pl.DataFrame({"word": ["the", "cat", "dog"], "ref": [100, 50, 25], "cmp": [20, 30, 80]})


@pytest.fixture
def df_equal() -> pl.DataFrame:
    """A mock corpus frame where both corpora are identical, so all contributions are zero."""
    return pl.DataFrame({"word": ["the", "cat", "dog"], "ref": [100, 50, 25], "cmp": [100, 50, 25]})


class TestNormalizeMethod:
    def test_default_normalized_by_is_none(self) -> None:
        """An unnormalized comparison has no `_normalized_by`."""
        assert Comparison("ref", "cmp")._state.normalized_by is None

    def test_normalizing_before_scoring_is_allowed(self) -> None:
        """Normalization has its own plan, so it does not need a score plan to exist yet."""
        assert comp("ref", "cmp").normalize("sum_abs")._state.normalized_by == "sum_abs"

    def test_resolving_without_scoring_raises(self) -> None:
        """A comparison that is only normalized has no scores to normalize."""
        with pytest.raises(ValueError, match="scored"):
            comp("ref", "cmp").normalize("sum_abs")._state.resolve()

    def test_raises_on_invalid_by(self) -> None:
        """`by` must be one of the two supported strategies."""
        scored = comp("ref", "cmp").score.proportion()
        with pytest.raises(ValueError, match="sum_abs"):
            scored.normalize("bad")  # type: ignore[arg-type]

    def test_proportion_abs_sum_raises_eagerly(self) -> None:
        """`abs_sum` on a proportion comparison is zero by construction, caught at call time."""
        scored = comp("ref", "cmp").score.proportion()
        with pytest.raises(ValueError, match="proportion"):
            scored.normalize("abs_sum")

    def test_sets_state_and_preserves_metadata(self) -> None:
        """`normalize` returns a new comparison recording `_normalized_by` and keeping metadata."""
        scored = comp("ref", "cmp").score.proportion()
        normalized = scored.normalize("sum_abs")

        assert normalized is not scored
        assert normalized._state.normalized_by == "sum_abs"
        assert scored._state.normalized_by is None
        assert normalized._state.method == scored._state.method
        assert normalized._state.is_weighted_avg == scored._state.is_weighted_avg
        assert normalized._state.attributed_by == scored._state.attributed_by

    def test_renormalizing_same_strategy_is_noop(self) -> None:
        """Normalizing again by the same strategy returns the comparison unchanged."""
        normalized = comp("ref", "cmp").score.proportion().normalize("sum_abs")

        assert normalized.normalize("sum_abs") is normalized

    def test_renormalizing_different_strategy_raises(self) -> None:
        """Normalizing by a different strategy than the existing one is an error."""
        normalized = comp("ref", "cmp").score.shannon_entropy().normalize("sum_abs")

        with pytest.raises(ValueError, match="already normalized"):
            normalized.normalize("abs_sum")

    def test_alias_preserves_normalized_by(self) -> None:
        """Aliasing carries the normalization state forward."""
        normalized = comp("ref", "cmp").score.proportion().normalize("sum_abs")
        assert normalized.alias("foo")._state.normalized_by == "sum_abs"

    def test_normalizing_before_scoring_matches_normalizing_after(
        self, df_mock: pl.DataFrame
    ) -> None:
        """Normalization occupies its own plan, so the order it is called in does not matter."""
        before = comp("ref", "cmp").normalize("sum_abs").score.shannon_entropy()
        after = comp("ref", "cmp").score.shannon_entropy().normalize("sum_abs")

        assert_frame_equal(
            before._state.resolve()(df_mock, "word").lazy().collect(),
            after._state.resolve()(df_mock, "word").lazy().collect(),
        )

    def test_invalid_normalization_for_a_later_scoring_method_raises(self) -> None:
        """The scoring method is only known to be compatible once every plan is in place."""
        normalized = comp("ref", "cmp").normalize("abs_sum")

        with pytest.raises(ValueError, match="proportion"):
            normalized.score.proportion()._state.resolve()


class TestNormalizedPlan:
    def test_normalization_constant_is_the_real_divisor(self, df_mock: pl.DataFrame) -> None:
        """The constant column holds the divisor computed from the un-normalized measure scores."""
        raw = comp("ref", "cmp").score.shannon_entropy()._state.resolve()(df_mock, "word")
        expected = raw[SCORE_COL].abs().sum()

        normalized = (
            comp("ref", "cmp")
            .score.shannon_entropy()
            .normalize("sum_abs")
            ._state.resolve()(df_mock, "word")
            .lazy()
            .collect()
        )

        assert normalized[NORMALIZATION_CONSTANT_COL].unique().item() == pytest.approx(expected)

    def test_sum_abs_equals_abs_sum_for_nonnegative_measure(self, df_mock: pl.DataFrame) -> None:
        """For an inherently non-negative measure the two strategies coincide."""
        sum_abs = (
            comp("ref", "cmp")
            .score.jensen_shannon_divergence()
            .normalize("sum_abs")
            ._state.resolve()(df_mock, "word")
            .lazy()
            .collect()
        )
        abs_sum = (
            comp("ref", "cmp")
            .score.jensen_shannon_divergence()
            .normalize("abs_sum")
            ._state.resolve()(df_mock, "word")
            .lazy()
            .collect()
        )
        assert_series_equal(sum_abs[SCORE_COL], abs_sum[SCORE_COL])

    @pytest.mark.parametrize(
        "score_func, by",
        [
            (lambda s: s.frequency(), "sum_abs"),
            (lambda s: s.proportion(), "sum_abs"),
            (lambda s: s.shannon_entropy(), "sum_abs"),
            (lambda s: s.jensen_shannon_divergence(), "sum_abs"),
            (lambda s: s.frequency(), "abs_sum"),
        ],
    )
    def test_zero_divisor_raises_when_planned(
        self, df_equal: pl.DataFrame, score_func: Callable[[Scorer], Comparison], by: str
    ) -> None:
        """A data-dependent zero divisor is caught when the plan runs (mirrors KLD validation)."""
        normalized = score_func(comp("ref", "cmp").score).normalize(by)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="zero"):
            normalized._state.resolve()(df_equal, "word")
