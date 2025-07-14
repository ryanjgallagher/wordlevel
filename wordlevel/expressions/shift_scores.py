import polars as pl

from wordlevel.expressions.utils import normalize_freqs, mixed_distribution
from wordlevel.expressions.entropy import tsallis_entropy_weighted_avg_scores


def proportion_shift_scores(
    freqs1: pl.Expr, 
    freqs2: pl.Expr,
    alias1: str | None=None,
    alias2: str | None=None
) -> tuple[pl.Expr, pl.Expr]:
    """
    """
    return pl.lit(1).alias(alias1), pl.lit(1).alias(alias2)

def tsallis_entropy_shift_scores(
    freqs1: pl.Expr, 
    freqs2: pl.Expr, 
    alpha: float=1,
    base: float=2,
    normalize: bool=True,
    alias1: str | None=None,
    alias2: str | None=None
) -> tuple[pl.Expr, pl.Expr]:
    """
    """
    scores1 = tsallis_entropy_weighted_avg_scores(
        freqs1, alpha=alpha, base=base, normalize=normalize
    )
    scores2 = tsallis_entropy_weighted_avg_scores(
        freqs2, alpha=alpha, base=base, normalize=normalize
    )

    return scores1.alias(alias1), scores2.alias(alias2)

def kullback_leibler_divergence_shift_scores(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    base: float=2,
    normalize: bool=True,
    alias1: str | None=None,
    alias2: str | None=None
) -> tuple[pl.Expr, pl.Expr]:
    """
    KLD = p * log(p / q)
        = p * log p - p * log q
        = -p * log q - (-p * log p)

    Let p = freqs1, q = freqs2, and have the scores correspond accordingly, i.e scores1 = -log p, 
    scores2 = -log q
    """
    scores1,scores2 = tsallis_entropy_shift_scores(
        freqs1, freqs2, alpha=1, base=base, normalize=normalize, alias1=alias1, alias2=alias2
    )

    return scores1.alias(alias1), scores2.alias(alias2)

def jensen_shannon_divergence_shift_scores(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    alpha: float=1,
    base: float=2,
    weight1: float=0.5,
    weight2: float=0.5,
    normalize: bool=True,
    alias1: str | None=None,
    alias2: str | None=None
) -> tuple[pl.Expr, pl.Expr]:
    """
    """
    probs1 = normalize_freqs(freqs1, normalize=normalize)
    probs2 = normalize_freqs(freqs2, normalize=normalize)
    
    if weight1 + weight2 != 1:
        weight1,weight2 = weight1 / (weight1 + weight2), weight2 / (weight1 + weight2)
    
    probs_mixed = mixed_distribution(
        probs1, probs2, weight1=weight1, weight2=weight2, normalize=False
    )

    if alpha == 1:
        scores1 = weight1 * (probs_mixed.log(base=base) - probs1.log(base=base))
        scores2 = weight2 * (probs2.log(base=base) - probs_mixed.log(base=base))
    else:
        scores1 = weight1 * (probs_mixed.pow(alpha - 1) - probs1.pow(alpha - 1)) / (alpha - 1)
        scores2 = weight2 * (probs2.pow(alpha - 1) - probs_mixed.pow(alpha - 1)) / (alpha - 1)

    return scores1.alias(alias1), scores2.alias(alias2)