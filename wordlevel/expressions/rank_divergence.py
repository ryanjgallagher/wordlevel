import polars as pl
from math import isinf


def rank(expr: pl.Expr, alpha: float=1) -> pl.Expr:
    """
    """
    if alpha < 0:
        raise ValueError('alpha must be non-negative')
    
    ranks = expr.rank(method='average', descending=True)
    
    if (alpha > 0) and (not isinf(alpha)):
        ranks = ranks.pow(alpha)

    return ranks

def rank_divergence(freqs1: pl.Expr, freqs2: pl.Expr, alpha: float=1.0) -> pl.Expr:
    """
    """
    summands = rank_divergence_summands(freqs1, freqs2, alpha=alpha)
    norm = rank_divergence_norm_summands(freqs1, freqs2, alpha=alpha).sum()

    return summands / norm

def rank_divergence_summands(freqs1: pl.Expr, freqs2: pl.Expr, alpha: float=1.0) -> pl.Expr:
    """
    """
    ranks1 = rank(freqs1, alpha=alpha)
    ranks2 = rank(freqs2, alpha=alpha)

    if alpha == 0:
        summands = (ranks1.log() - ranks2.log()).abs()
    elif isinf(alpha):
        summands = pl.when(ranks1 == ranks2).then(0).otherwise(pl.max_horizontal(1 / ranks1, 1 / ranks2))
    else:
        prefactor = (alpha + 1.0) / alpha
        summands = prefactor * (1 / ranks1 - 1 / ranks2).abs().pow(1.0 / (alpha + 1))

    return summands

def rank_divergence_norm_summands(freqs1: pl.Expr, freqs2: pl.Expr, alpha: float=1.0) -> pl.Expr:
    """
    """
    n_types1 = freqs1.filter(freqs1 > 0).len()
    n_types2 = freqs2.filter(freqs2 > 0).len()

    comp1 = rank_divergence_norm_summand_component(freqs1, n_types1, n_types2, alpha)
    comp2 = rank_divergence_norm_summand_component(freqs2, n_types2, n_types1, alpha)

    return comp1 + comp2

def rank_divergence_norm_summand_component(
        freqs: pl.Expr, 
        n_types1: int, 
        n_types2: int, 
        alpha: float=1.0
    ) -> pl.Expr:
    """
    """
    ranks = rank(freqs, alpha=alpha)

    if alpha == 0:
        summand = (ranks.log() - (n_types1 + 0.5 * n_types2 + 0.5).log()).abs()
    elif isinf(alpha):
        summand = 1 / ranks
    else:
        summand = (
            ((alpha + 1.0) / alpha) 
            * (1 / ranks - 1 / (n_types1 + 0.5 * n_types2 + 0.5) ** alpha)
            .abs()
            .pow(1.0 / (alpha + 1))
        )

    component = pl.when(freqs > 0).then(summand).otherwise(0)

    return component
