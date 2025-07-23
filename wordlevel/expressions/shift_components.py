import polars as pl

from wordlevel.expressions.utils import normalize_freqs


def frequency_diff(freqs1: pl.Expr, freqs2: pl.Expr, normalize: bool=True) -> pl.Expr:
    """
    """
    probs1 = normalize_freqs(freqs1, normalize=normalize)
    probs2 = normalize_freqs(freqs2, normalize=normalize)

    return probs2 - probs1 

def weighted_score_diff(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    scores1: pl.Expr,
    scores2: pl.Expr,
    normalize: bool=True
) -> pl.Expr:
    """
    """
    probs1 = normalize_freqs(freqs1, normalize=normalize)
    probs2 = normalize_freqs(freqs2, normalize=normalize)

    return (0.5 * probs1 + 0.5 * probs2) * (scores2 - scores1)

def reference_score_diff(
    scores1: pl.Expr,
    scores2: pl.Expr,
    reference_score: float
) -> pl.Expr:
    """
    """
    return (0.5 * scores1 + 0.5 * scores2) - reference_score