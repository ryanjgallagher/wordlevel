from typing import Callable

import pytest
import polars as pl
from polars import col
from polars.testing import assert_series_equal

from wordlevel.api.comparison.comparison import Comparison, comp
from wordlevel.api.comparison.score import Scorer
from wordlevel.constants import SCORE_COL
from wordlevel.core.utils import normalize_freqs


@pytest.fixture
def df_mock() -> pl.DataFrame:
    """A mock corpus frame with the normalized-frequency columns the plans expect."""
    return pl.DataFrame(
        {"word": ["the", "cat", "dog"], "ref": [100, 50, 25], "cmp": [20, 30, 80]}
    ).with_columns(
        normalize_freqs(col("ref")).alias("ref__wl_normed"),
        normalize_freqs(col("cmp")).alias("cmp__wl_normed"),
    )


@pytest.fixture
def df_equal() -> pl.DataFrame:
    """A mock corpus frame where both corpora are identical, so all contributions are zero."""
    return pl.DataFrame(
        {"word": ["the", "cat", "dog"], "ref": [100, 50, 25], "cmp": [100, 50, 25]}
    ).with_columns(
        normalize_freqs(col("ref")).alias("ref__wl_normed"),
        normalize_freqs(col("cmp")).alias("cmp__wl_normed"),
    )


class TestNormalizeMethod:
    def test_default_normalized_by_is_none(self) -> None:
        """An unnormalized comparison has no `_normalized_by`."""
        assert Comparison("ref", "cmp")._normalized_by is None

    def test_raises_if_not_scored(self) -> None:
        """Normalizing before scoring has no plan to compose onto."""
        with pytest.raises(ValueError, match="scored"):
            comp("ref", "cmp").normalize("sum_abs")

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
        assert normalized._normalized_by == "sum_abs"
        assert scored._normalized_by is None
        assert normalized._method == scored._method
        assert normalized._is_weighted_avg == scored._is_weighted_avg
        assert normalized._attributed_by == scored._attributed_by

    def test_renormalizing_does_not_raise(self) -> None:
        """Re-normalizing is redundant computation, not an error."""
        twice = comp("ref", "cmp").score.proportion().normalize("sum_abs").normalize("sum_abs")
        assert twice._normalized_by == "sum_abs"

    def test_alias_preserves_normalized_by(self) -> None:
        """Aliasing carries the normalization state forward."""
        normalized = comp("ref", "cmp").score.proportion().normalize("sum_abs")
        assert normalized.alias("foo")._normalized_by == "sum_abs"


class TestNormalizedPlan:
    def test_sum_abs_equals_abs_sum_for_nonnegative_measure(self, df_mock: pl.DataFrame) -> None:
        """For an inherently non-negative measure the two strategies coincide."""
        sum_abs = (
            comp("ref", "cmp")
            .score.jensen_shannon_divergence()
            .normalize("sum_abs")
            ._plan(df_mock, "word", "__wl_normed")
        )
        abs_sum = (
            comp("ref", "cmp")
            .score.jensen_shannon_divergence()
            .normalize("abs_sum")
            ._plan(df_mock, "word", "__wl_normed")
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
            normalized._plan(df_equal, "word", "__wl_normed")
