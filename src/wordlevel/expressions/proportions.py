import polars as pl

from wordlevel.expressions.utils import alias_expressions


def proportion_dist_scores(freqs: pl.Expr) -> pl.Expr:
    """ """
    return pl.lit(1)


def proportion_dual_dist_scores(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    alias1: str | None = None,
    alias2: str | None = None,
) -> tuple[pl.Expr, pl.Expr]:
    """ """
    return alias_expressions(
        proportion_dist_scores(freqs1), proportion_dist_scores(freqs2), alias1, alias2
    )


def proportion_clean_dual_dist_scores(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    scores1: pl.Expr,
    scores2: pl.Expr,
    alias1: str | None = None,
    alias2: str | None = None,
) -> tuple[pl.Expr, pl.Expr]:
    """Clean proportion scores - just returns the scores as-is since proportions are always 1."""
    return alias_expressions(scores1, scores2, alias1, alias2)
