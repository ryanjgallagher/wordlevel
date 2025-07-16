import polars as pl

from wordlevel.expressions.shift_scores import jensen_shannon_divergence_shift_scores
from wordlevel.expressions.utils import normalize_freqs


def jensen_shannon_divergence(
    freqs1: pl.Expr, 
    freqs2: pl.Expr, 
    weight1: float=0.5, 
    weight2: float=0.5,
    base: int=2,
    alpha: float=1,
    normalize: bool=True
) -> pl.Expr:
    """
    """
    probs1 = normalize_freqs(freqs1, normalize=normalize)
    probs2 = normalize_freqs(freqs2, normalize=normalize)

    scores1,scores2 = jensen_shannon_divergence_shift_scores(
        probs1, probs2, alpha=alpha, base=base, weight1=weight1, weight2=weight2, normalize=False
    )
    scores1 = pl.when(probs1 > 0).then(scores1).otherwise(0)
    scores2 = pl.when(probs2 > 0).then(scores2).otherwise(0)

    jsd = probs2 * scores2 - probs1 * scores1

    return jsd