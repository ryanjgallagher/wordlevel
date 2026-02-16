from dataclasses import dataclass
from functools import cached_property
from typing_extensions import Self

import polars as pl

from wordlevel.types import PolarsFrame
from wordlevel.compare.constants import NORMALIZATION_SUFFIX
from wordlevel.compare.comparisons.score import Scorer
from wordlevel.compare.comparisons.types import ScorePlan


class Comparison:
    """ """

    def __init__(
        self,
        reference: str,
        comparison: str,
        alias: str | None = None,
        method: str | None = None,
        is_weighted_avg: bool | None = None,
        plan: ScorePlan | None = None,
        normalization_suffix: str = NORMALIZATION_SUFFIX,
        parameters: dict | None = None,
    ):
        self._ref = reference
        self._cmp = comparison
        self._alias = alias
        self._method = method
        self._is_weighted_avg = is_weighted_avg
        self._plan = plan
        self._normalization_suffix = normalization_suffix

        if parameters is None:
            self._parameters = dict()
        else:
            self._parameters = parameters.copy()

    @property
    def _ref_normed(self) -> str:
        return f"{self._ref}{self._normalization_suffix}"

    @property
    def _cmp_normed(self) -> str:
        return f"{self._cmp}{self._normalization_suffix}"

    @cached_property
    def score(self) -> Scorer:
        """ """
        return Scorer(self)

    def alias(self, name: str) -> Self:
        """Return a new Comparison with the specified alias."""
        return self.__class__(
            reference=self._ref,
            comparison=self._cmp,
            alias=name,
            method=self._method,
            is_weighted_avg=self._is_weighted_avg,
            plan=self._plan,
            normalization_suffix=self._normalization_suffix,
            parameters=self._parameters,
        )

    def _with_normalization_suffix(self, suffix: str) -> Self:
        """Return a new Comparison with the specified normalization suffix."""
        return self.__class__(
            reference=self._ref,
            comparison=self._cmp,
            alias=self._alias,
            method=self._method,
            is_weighted_avg=self._is_weighted_avg,
            plan=self._plan,
            normalization_suffix=suffix,
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
    parameters: dict

    @property
    def df(self) -> pl.DataFrame:
        df = self._lf.collect()
        object.__setattr__(self, "_lf", df.lazy())

        return df

    @classmethod
    def from_comparison(cls, comparison: Comparison, df: PolarsFrame) -> Self:
        """ """
        if comparison._plan is None:
            raise ValueError("Cannot construct a comparison result without a plan")
        if comparison._alias is None:
            raise ValueError("Cannot construct a comparison result without an alias")
        # fmt: off
        if (comparison._is_weighted_avg is None) or (comparison._parameters is None) or (comparison._method is None):
            raise ValueError("Missing comparison metadata: `method`, `parameters`, or `is_weighted_avg` is None")
        # fmt: on

        return cls(
            _lf=comparison._plan(df).lazy(),
            reference=comparison._ref,
            comparison=comparison._cmp,
            alias=comparison._alias,
            method=comparison._method,
            is_weighted_avg=comparison._is_weighted_avg,
            parameters=comparison._parameters.copy(),
        )
