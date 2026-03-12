from typing import Protocol

import polars as pl


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
