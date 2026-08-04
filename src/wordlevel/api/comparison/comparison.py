from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

import polars as pl
from typing_extensions import Self

from wordlevel.api.comparison.exclude import ComparisonExcluder
from wordlevel.api.comparison.normalization import make_normalized_plan
from wordlevel.api.comparison.score import Scorer
from wordlevel.api.comparison.state import ComparisonState
from wordlevel.constants import NORMALIZATION_CONSTANT_COL
from wordlevel.core.comparison.normalization import NORMALIZE_BY
from wordlevel.core.comparison.score.attribution import ATTRIBUTION_LABELS, ATTRIBUTION_TYPE


class Comparison:
    """
    A comparison between two corpora: a reference corpus and a comparison corpus.

    Args:
        reference:
            Name of the reference corpus
        comparison:
            Name of the comparison corpus
    """

    def __init__(self, reference: str, comparison: str):
        self._state = ComparisonState(reference=reference, comparison=comparison)

    def _update_state(self, **changes: Any) -> Comparison:
        """
        Returns a new `Comparison` with the given state fields replaced.

        Args:
            changes:
                The [`ComparisonState`][wordlevel.api.comparison.state.ComparisonState] fields
                to replace, and the values to replace them with
        """
        comparison = self.__class__(self._state.reference, self._state.comparison)
        comparison._state = replace(self._state, **changes)

        return comparison

    @cached_property
    def score(self) -> Scorer:
        """
        Namespace for scoring comparisons between the reference and comparison corpora.

        See the [available scoring methods](score/index.md).
        """
        return Scorer(self)

    @cached_property
    def exclude(self) -> ComparisonExcluder:
        """
        Namespace for excluding tokens from the reference and comparison corpora.

        See the [available exclusion strategies](exclude/index.md).
        """
        return ComparisonExcluder(self)

    def alias(self, name: str) -> Comparison:
        """
        Returns a new `Comparison` with the given alias, an internal name for it.

        Example:
            ```python
            --8<-- "snippets/comparison/comparison.py:alias"
            ```

        Args:
            name: The name to give to the comparison

        Returns:
            A new `Comparison` with the given alias
        """
        return self._update_state(alias=name)

    def normalize(self, by: NORMALIZE_BY) -> Comparison:
        r"""
        Returns a new `Comparison` whose token contributions are normalized onto a common scale.

        Each scoring measure produces a per-token contribution $\delta_\tau$. Normalization
        rescales every contribution by a single constant so that scores become interpretable
        and comparable across measures. There are two normalization strategies:

        - `"sum_abs"` divides by the sum of absolute contributions, $\sum_\tau |\delta_\tau|$. The
          absolute values of the normalized contributions then sum to 1.
        - `"abs_sum"` divides by the absolute sum of contributions, $|\sum_\tau \delta_\tau|$. The
          normalized contributions then sum to $\pm 1$.

        Normalizing again by the same strategy is a no-op; normalizing by a different strategy
        raises an error, since the contributions are already on a normalized scale.

        Example:
            ```python
            --8<-- "snippets/comparison/normalization.py:setup"

            --8<-- "snippets/comparison/normalization.py:normalize"
            ```

        Warning:
            Using this method forces the comparison to be evaluated eagerly rather than lazily,
            because validating it requires computing the scores over the full vocabulary.

        Args:
            by:
                The normalization strategy. `"sum_abs"` divides every token's contribution by
                $\sum_\tau |\delta_\tau|$; `"abs_sum"` divides by $|\sum_\tau \delta_\tau|$

        Returns:
            A new `Comparison` with normalized token contributions

        Raises:
            ValueError:
                If the comparison is already normalized by a different strategy.
            ValueError: If `by` is not a recognized normalization strategy.
            ValueError:
                If `by` is `"abs_sum"` for a `proportion` comparison, whose contributions sum to
                zero by construction, so the absolute sum is always zero.
        """
        if self._state.normalized_by is not None:
            if self._state.normalized_by == by:
                return self
            raise ValueError(
                f"Comparison is already normalized by {self._state.normalized_by!r}; cannot also "
                f"normalize by {by!r}."
            )

        self._state.validate_normalization(by)

        return self._update_state(
            score_normalization_plan=make_normalized_plan(by),
            normalized_by=by,
        )


def comp(reference: str, comparison: str) -> Comparison:
    """
    Convenience constructor for a
    [`Comparison`][wordlevel.api.comparison.comparison.Comparison].

    Equivalent to `Comparison(reference=reference, comparison=comparison)`.

    Example:
        ```python
        --8<-- "snippets/comparison/comparison.py:comp"
        ```

    Args:
        reference:
            Name of the reference corpus
        comparison:
            Name of the comparison corpus

    Returns:
        A new `Comparison` between the reference and comparison corpora
    """
    return Comparison(reference=reference, comparison=comparison)


@dataclass(frozen=True)
class ComparisonResult:
    """
    The result of a [`Comparison`][wordlevel.api.comparison.comparison.Comparison] when a plan is
    applied to corpus data.

    Attributes:
        df:
            The dataframe containing all scoring information for the comparison. Contains one row
            per token, with columns for the score, rank, attribution, and per-token contribution
            components.
        reference:
            Name of the reference corpus
        comparison:
            Name of the comparison corpus
        alias:
            The name of the comparison
        method:
            The name of a scoring method
        is_weighted_avg:
            Whether the scoring method can be interpreted as a weighted average or difference in
            weighted averages
        is_nonnegative:
            Whether the scoring method is inherently non-negative, i.e. all token-level
            contributionso are non-negative
        attributed_by:
            The attribution type used to split token contributions between the two bars of a bar
            plot. `"frequency"` and `"rank"` attribute each token to a corpus; `"score_sign"`
            attributes each token by whether it reinforces or offsets the overall difference
        attribution_labels:
            The pair of labels the attribution produces, determined by `attributed_by`
        parameters:
            Any parameters specfic to the scoring method when making the comparison
        exclusions:
            The exclusions applied to the vocabulary, keyed by the kind of exclusion. Each value
            describes what that kind of exclusion removed, e.g. `{"tokens": {"the", "a"}}`
        normalized:
            Whether the token contributions have been normalized
        normalized_by:
            The normalization strategy applied to the token contributions, or `None` if they are not
            normalized
        normalization_constant:
            The constant by which the token contributions were divided to normalize them, or `None`
            if they are not normalized
    """

    _lf: pl.LazyFrame
    reference: str
    comparison: str
    alias: str
    method: str
    is_weighted_avg: bool
    is_nonnegative: bool
    attributed_by: ATTRIBUTION_TYPE
    parameters: dict
    exclusions: dict[str, Any] = field(default_factory=dict)
    normalized_by: str | None = None

    @property
    def normalized(self) -> bool:
        """Whether the token contributions have been normalized."""
        return self.normalized_by is not None

    @property
    def normalization_constant(self) -> float | None:
        """
        The constant by which the token contributions were divided to normalize them.

        `None` if the contributions are not normalized.
        """
        if not self.normalized:
            return None

        constant = self.df[NORMALIZATION_CONSTANT_COL].first()
        if constant is None:
            return None

        return float(constant)  # type: ignore[arg-type]

    @property
    def attribution_labels(self) -> tuple[str, str]:
        """
        The pair of labels the attribution produces, determined by `attributed_by`.

        Corpus attribution names each token's bucket by corpus; score-sign attribution names it by
        whether the token reinforces or offsets the overall difference.
        """
        return ATTRIBUTION_LABELS[self.attributed_by]

    @property
    def df(self) -> pl.DataFrame:
        """
        The dataframe containing all scoring information for the comparison.

        Contains one row per token, with columns for the score, rank, attribution, and per-token
        contribution components.
        """
        df = self._lf.collect()
        object.__setattr__(self, "_lf", df.lazy())

        return df

    @classmethod
    def from_comparison(
        cls,
        comparison: Comparison,
        df: pl.DataFrame | pl.LazyFrame,
        token_col: str,
    ) -> Self:
        """
        Creates a `ComparisonResult` from a `Comparison` by applying the comparison's plan to the
        corpus data in a catalog.

        Args:
            comparison:
                The comparison to score
            df:
                The dataframe representing the corpus data to score
            token_col:
                Name of the column indicating tokens in `df`

        Returns:
            A new `ComparisonResult` with the scored data

        Raises:
            ValueError: If the comparison has no scoring plan.
            ValueError: If the comparison has no alias.
            ValueError: If any of the comparison's scoring metadata is missing.
        """
        state = comparison._state
        resolved_plan = state.resolve()

        if state.alias is None:
            raise ValueError("Cannot construct a comparison result without an alias")
        if (
            state.method is None
            or state.is_weighted_avg is None
            or state.is_nonnegative is None
            or state.attributed_by is None
        ):
            raise ValueError(
                "Missing comparison metadata: `method`, `is_weighted_avg`, `is_nonnegative`, or "
                "`attributed_by` is None"
            )

        return cls(
            _lf=resolved_plan(df, token_col).lazy(),
            reference=state.reference,
            comparison=state.comparison,
            alias=state.alias,
            method=state.method,
            is_weighted_avg=state.is_weighted_avg,
            is_nonnegative=state.is_nonnegative,
            attributed_by=state.attributed_by,
            parameters=dict(state.parameters),
            exclusions=dict(state.exclusions),
            normalized_by=state.normalized_by,
        )
