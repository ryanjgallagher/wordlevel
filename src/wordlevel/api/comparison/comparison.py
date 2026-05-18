from dataclasses import dataclass
from functools import cached_property
from typing_extensions import Self

import polars as pl

from wordlevel.api.comparison.score import Scorer, ScorePlan


class Comparison:
    """ """

    def __init__(
        self,
        reference: str,
        comparison: str,
        alias: str | None = None,
        method: str | None = None,
        is_weighted_avg: bool | None = None,
        is_nonnegative: bool | None = None,
        plan: ScorePlan | None = None,
        parameters: dict | None = None,
    ):
        self._ref = reference
        self._cmp = comparison
        self._alias = alias
        self._method = method
        self._is_weighted_avg = is_weighted_avg
        self._is_nonnegative = is_nonnegative
        self._plan = plan

        if parameters is None:
            self._parameters = dict()
        else:
            self._parameters = parameters.copy()

    @cached_property
    def score(self) -> Scorer:
        """
        The scorer
        """
        return Scorer(self)

    def alias(self, name: str) -> Self:
        """Return a new Comparison with the specified alias."""
        return self.__class__(
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
    """ """
    return Comparison(reference=reference, comparison=comparison)


@dataclass(frozen=True)
class ComparisonResult:
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
        """ """
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
