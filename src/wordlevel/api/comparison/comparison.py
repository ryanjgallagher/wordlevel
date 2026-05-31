from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing_extensions import Self

import polars as pl

from wordlevel.api.comparison.score import Scorer, ScorePlan


class Comparison:
    """
    A comparison between two corpora: a reference and a comparison.

    Args:
        reference:
            Name of the reference corpus
        comparison:
            Name of the comparison corpus
    """

    def __init__(self, reference: str, comparison: str):
        self._ref = reference
        self._cmp = comparison
        self._alias: str | None = None
        self._method: str | None = None
        self._is_weighted_avg: bool | None = None
        self._is_nonnegative: bool | None = None
        self._plan: ScorePlan | None = None
        self._parameters: dict = {}

    @classmethod
    def _with_state(
        cls,
        reference: str,
        comparison: str,
        *,
        alias: str | None = None,
        method: str | None = None,
        is_weighted_avg: bool | None = None,
        is_nonnegative: bool | None = None,
        plan: ScorePlan | None = None,
        parameters: dict | None = None,
    ) -> Comparison:
        """
        Constructs a `Comparison` with parameters related to its state.

        Args:
            alias:
                The name of the comparison
            method:
                The name of a scoring method
            is_weighted_avg
                Whether the scoring method can be interpreted as a weighted average or difference in
                weighted averages
            is_nonnegative
                Whether the scoring method is inherently non-negative, i.e. all token-level
                contributionso are non-negative
            plan:
                The "scoring plan" describing how to compare the reference and comparison corpora
                according to thes scoring method
            parameters:
                Any parameters specfic to the scoring method when making the comparison
        """
        instance = cls(reference, comparison)
        instance._alias = alias
        instance._method = method
        instance._is_weighted_avg = is_weighted_avg
        instance._is_nonnegative = is_nonnegative
        instance._plan = plan
        instance._parameters = (parameters or {}).copy()

        return instance

    @cached_property
    def score(self) -> Scorer:
        """
        Namespace for scoring comparisons between the reference and comparison corpora.

        See the [available scoring methods](score/index.md).
        """
        return Scorer(self)

    def alias(self, name: str) -> Comparison:
        """
        Returns a new `Comparison` with the given alias, a name for the comparison.

        Args:
            name: The name to give to the comparison

        Returns:
            A new `Comparison` with the given alias
        """
        return self._with_state(
            reference=self._ref,
            comparison=self._cmp,
            alias=name,
            method=self._method,
            is_weighted_avg=self._is_weighted_avg,
            is_nonnegative=self._is_nonnegative,
            plan=self._plan,
            parameters=self._parameters,
        )


def comp(reference: str, comparison: str) -> Comparison:
    """
    Convenience constructor for a
    [`Comparison`][wordlevel.api.comparison.comparison.Comparison].

    Equivalent to `Comparison(reference=reference, comparison=comparison)`.

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
        reference:
            Name of the reference corpus
        comparison:
            Name of the comparison corpus
        alias:
            The name of the comparison
        method:
            The name of a scoring method
        is_weighted_avg
            Whether the scoring method can be interpreted as a weighted average or difference in
            weighted averages
        is_nonnegative
            Whether the scoring method is inherently non-negative, i.e. all token-level
            contributionso are non-negative
        plan:
            The "scoring plan" describing how to compare the reference and comparison corpora
            according to thes scoring method
        parameters:
            Any parameters specfic to the scoring method when making the comparison
    """

    _lf: pl.LazyFrame
    reference: str
    comparison: str
    alias: str
    method: str
    is_weighted_avg: bool
    is_nonnegative: bool
    parameters: dict

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
        normalization_suffix: str,
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
            normalization_suffix:
                Suffix appended to normalized frequency column names

        Returns:
            A new `ComparisonResult` with the scored data

        Raises:
            ValueError: If the comparison has no scoring plan.
            ValueError: If the comparison has no alias.
            ValueError: If any of the comparison's scoring metadata is missing.
        """
        if comparison._plan is None:
            raise ValueError("Cannot construct a comparison result without a plan")
        if comparison._alias is None:
            raise ValueError("Cannot construct a comparison result without an alias")
        if (
            comparison._method is None
            or comparison._is_weighted_avg is None
            or comparison._is_nonnegative is None
            or comparison._parameters is None
        ):
            raise ValueError(
                "Missing comparison metadata: `method`, `parameters`, `is_weighted_avg`, or "
                "`is_nonnegative` is None"
            )

        return cls(
            _lf=comparison._plan(df, token_col, normalization_suffix).lazy(),
            reference=comparison._ref,
            comparison=comparison._cmp,
            alias=comparison._alias,
            method=comparison._method,
            is_weighted_avg=comparison._is_weighted_avg,
            is_nonnegative=comparison._is_nonnegative,
            parameters=comparison._parameters.copy(),
        )
