import polars as pl
from math import e
from wordlevel.expressions.utils import normalize_freqs


def tsallis_entropy_weighted_avg_scores(
    freqs: pl.Expr, 
    alpha: float=1, 
    base: float=2, 
    normalize: bool=True
) -> pl.Expr:
    """
    """
    probs = normalize_freqs(freqs, normalize=normalize)

    if alpha == 1:
        scores = -1 * probs.log(base=base)
    else:
        scores = (1 / (alpha - 1)) * (1 - probs.pow(alpha - 1))

    return scores

def tsallis_entropy(freqs: pl.Expr, alpha: float=1, base: float=2, normalize: bool=True) -> pl.Expr:
    """
    """
    probs = normalize_freqs(freqs, normalize=normalize)

    if alpha == 1:
        scores = (
            pl.when(probs > 0)
            .then(tsallis_entropy_weighted_avg_scores(probs, base=base, normalize=False))
            .otherwise(0)
        )
    else:
        scores = tsallis_entropy_weighted_avg_scores(probs, alpha=alpha, normalize=False)

    entropies = probs * scores

    return entropies


def shannon_entropy(freqs: pl.Expr, base: float=2, normalize: bool=True) -> pl.Expr:
    """
    """
    return tsallis_entropy(freqs, alpha=1, base=base, normalize=normalize)
