from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_series_equal

from wordlevel.api.comparison.comparison import Comparison, comp
from wordlevel.api.comparison.score import Scorer
from wordlevel.constants import NORMALIZATION_CONSTANT_COL, SCORE_COL


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
        assert comp("ref", "cmp").normalize("total_diff")._state.normalized_by == "total_diff"

    def test_raises_on_invalid_by(self) -> None:
        """`by` must be one of the two supported strategies."""
        scored = comp("ref", "cmp").score.proportion()
        with pytest.raises(ValueError, match="total_diff"):
            scored.normalize("bad")  # type: ignore[arg-type]

    def test_proportion_net_diff_raises_eagerly(self) -> None:
        """`net_diff` on a proportion comparison is zero by construction, caught at call time."""
        scored = comp("ref", "cmp").score.proportion()
        with pytest.raises(ValueError, match="proportion"):
            scored.normalize("net_diff")

    def test_sets_state_and_preserves_metadata(self) -> None:
        """`normalize` returns a new comparison recording `_normalized_by` and keeping metadata."""
        scored = comp("ref", "cmp").score.proportion()
        normalized = scored.normalize("total_diff")

        assert normalized is not scored
        assert normalized._state.normalized_by == "total_diff"
        assert scored._state.normalized_by is None
        assert normalized._state.method == scored._state.method
        assert normalized._state.is_weighted_avg == scored._state.is_weighted_avg
        assert normalized._state.attributed_by == scored._state.attributed_by

    def test_renormalizing_same_strategy_is_noop(self) -> None:
        """Normalizing again by the same strategy returns the comparison unchanged."""
        normalized = comp("ref", "cmp").score.proportion().normalize("total_diff")

        assert normalized.normalize("total_diff") is normalized

    def test_renormalizing_different_strategy_raises(self) -> None:
        """Normalizing by a different strategy than the existing one is an error."""
        normalized = comp("ref", "cmp").score.shannon_entropy().normalize("total_diff")

        with pytest.raises(ValueError, match="already normalized"):
            normalized.normalize("net_diff")

    def test_alias_preserves_normalized_by(self) -> None:
        """Aliasing carries the normalization state forward."""
        normalized = comp("ref", "cmp").score.proportion().normalize("total_diff")
        assert normalized.alias("foo")._state.normalized_by == "total_diff"


class TestNormalizedPlan:
    def test_normalization_constant_is_the_real_divisor(self, df_mock: pl.DataFrame) -> None:
        """The constant column holds the divisor computed from the un-normalized measure scores."""
        raw = comp("ref", "cmp").score.shannon_entropy()._state.resolve()(df_mock, "word")
        expected = raw[SCORE_COL].abs().sum()

        normalized = (
            comp("ref", "cmp")
            .score.shannon_entropy()
            .normalize("total_diff")
            ._state.resolve()(df_mock, "word")
            .lazy()
            .collect()
        )

        assert normalized[NORMALIZATION_CONSTANT_COL].unique().item() == pytest.approx(expected)

    def test_total_diff_equals_net_diff_for_nonnegative_measure(
        self, df_mock: pl.DataFrame
    ) -> None:
        """For an inherently non-negative measure the two strategies coincide."""
        total_diff = (
            comp("ref", "cmp")
            .score.jensen_shannon_divergence()
            .normalize("total_diff")
            ._state.resolve()(df_mock, "word")
            .lazy()
            .collect()
        )
        net_diff = (
            comp("ref", "cmp")
            .score.jensen_shannon_divergence()
            .normalize("net_diff")
            ._state.resolve()(df_mock, "word")
            .lazy()
            .collect()
        )
        assert_series_equal(total_diff[SCORE_COL], net_diff[SCORE_COL])

    @pytest.mark.parametrize(
        "score_func, by",
        [
            (lambda s: s.frequency(), "total_diff"),
            (lambda s: s.proportion(), "total_diff"),
            (lambda s: s.shannon_entropy(), "total_diff"),
            (lambda s: s.jensen_shannon_divergence(), "total_diff"),
            (lambda s: s.frequency(), "net_diff"),
        ],
    )
    def test_zero_divisor_raises_when_planned(
        self, df_equal: pl.DataFrame, score_func: Callable[[Scorer], Comparison], by: str
    ) -> None:
        """A data-dependent zero divisor is caught when the plan runs (mirrors KLD validation)."""
        normalized = score_func(comp("ref", "cmp").score).normalize(by)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="zero"):
            normalized._state.resolve()(df_equal, "word")

    @pytest.mark.parametrize("check_nonzero", [True, False])
    def test_nonzero_divisor_normalizes_whether_or_not_checked(
        self, df_mock: pl.DataFrame, check_nonzero: bool
    ) -> None:
        """
        The corpora in `df_mock` differ, so the divisor is nonzero and the scores normalize whether
        or not the divisor is checked.
        """
        normalized = comp("ref", "cmp").score.frequency().normalize("total_diff", check_nonzero)

        df_normalized = normalized._state.resolve()(df_mock, "word").lazy().collect()

        assert df_normalized[SCORE_COL].abs().sum() == pytest.approx(1.0)

    def test_zero_divisor_does_not_raise_without_check_nonzero(
        self, df_equal: pl.DataFrame
    ) -> None:
        """
        The corpora in `df_equal` are identical, so the divisor is zero, which does not raise when
        it is not checked.
        """
        normalized = (
            comp("ref", "cmp").score.frequency().normalize("total_diff", check_nonzero=False)
        )

        normalized._state.resolve()(df_equal, "word").lazy().collect()


class TestCheckNonzeroLaziness:
    @pytest.fixture
    def lf_uncollectable(self) -> pl.LazyFrame:
        """A mock corpus frame whose strict cast of `"x"` fails only when it is collected."""
        return pl.LazyFrame(
            {"word": ["the", "cat"], "ref": ["100", "x"], "cmp": [20, 30]}
        ).with_columns(pl.col("ref").cast(pl.Int64))

    def test_check_nonzero_collects_the_plan(self, lf_uncollectable: pl.LazyFrame) -> None:
        """Checking for a zero divisor collects the plan, so the failing query runs."""
        normalized = (
            comp("ref", "cmp").score.frequency().normalize("total_diff", check_nonzero=True)
        )

        with pytest.raises(pl.exceptions.InvalidOperationError):
            normalized._state.resolve()(lf_uncollectable, "word")

    def test_plan_stays_lazy_without_check_nonzero(self, lf_uncollectable: pl.LazyFrame) -> None:
        """Without the check, the plan does not collect, so the failing query never runs."""
        normalized = (
            comp("ref", "cmp").score.frequency().normalize("total_diff", check_nonzero=False)
        )

        normalized._state.resolve()(lf_uncollectable, "word")
