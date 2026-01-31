from collections.abc import Callable
from typing import Protocol

import polars as pl

from wordlevel.types import PolarsFrame


ScorePlan = Callable[[PolarsFrame], PolarsFrame]


class ScoreFunc(Protocol):
    def __call__(
        self,
        freqs1: pl.Expr,
        freqs2: pl.Expr,
    ) -> pl.Expr: ...


class DualScoreFunc(Protocol):
    def __call__(
        self, freqs1: pl.Expr, freqs2: pl.Expr, alias1: str, alias2: str
    ) -> tuple[pl.Expr, pl.Expr]: ...


class DualCleanScoreFunc(Protocol):
    def __call__(
        self,
        freqs1: pl.Expr,
        freqs2: pl.Expr,
        scores1: pl.Expr,
        scores2: pl.Expr,
        alias1: str,
        alias2: str,
    ) -> tuple[pl.Expr, pl.Expr]: ...


class ComparisonInterface(Protocol):
    _ref: str
    _cmp: str
    _ref_normed: str
    _cmp_normed: str
    _normalization_suffix: str
    _alias: str | None
    _plan: ScorePlan | None

    def __init__(
        self,
        alias: str,
        reference: str,
        comparison: str,
        method: str,
        parameters: dict,
        is_weighted_avg: bool,
        plan: ScorePlan,
        normalization_suffix: str,
    ) -> None: ...
