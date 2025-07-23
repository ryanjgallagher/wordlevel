import polars as pl

from wordlevel.expressions.utils import alias_expressions


def proportion_shift_scores(
    freqs1: pl.Expr, 
    freqs2: pl.Expr,
    alias1: str | None=None,
    alias2: str | None=None
) -> tuple[pl.Expr, pl.Expr]:
    """
    """
    return alias_expressions(pl.lit(1), pl.lit(1), alias1, alias2)