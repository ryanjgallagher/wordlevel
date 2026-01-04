import polars as pl

from wordlevel.expressions.entropy import tsallis_entropy_dual_dist_scores
from wordlevel.expressions.utils import normalize_freqs, alias_expressions


def kullback_leibler_divergence_shift_scores(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    base: float = 2,
    normalize: bool = True,
    alias1: str | None = None,
    alias2: str | None = None,
) -> tuple[pl.Expr, pl.Expr]:
    """
    KLD = p * log(p / q)
        = p * log p - p * log q
        = -p * log q - (-p * log p)

    Let p = freqs1, q = freqs2, and have the scores correspond accordingly, i.e scores1 = -log p,
    scores2 = -log q
    """
    scores1, scores2 = tsallis_entropy_dual_dist_scores(
        freqs1, freqs2, alpha=1, base=base, normalize=normalize, alias1=alias1, alias2=alias2
    )

    return alias_expressions(scores1, scores2, alias1, alias2)


def kullback_leibler_divergence(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    base: int = 2,
    normalize: bool = True,
) -> pl.Expr:
    """
    KLD = p * log(p / q)
        = p * log p - p * log q
        = -p * log q - (-p * log p)
        = p * log(1/q) - p * log(1/p)

    Let p = freqs1, q = freqs2. That is, p is the reference distribution ("true" distribution)
    """
    probs1 = normalize_freqs(freqs1, normalize=normalize)
    probs2 = normalize_freqs(freqs2, normalize=normalize)

    scores1, scores2 = kullback_leibler_divergence_shift_scores(
        probs1, probs2, base=base, normalize=False
    )
    # scores1 follows the normal entropy limiting: if probs1 = 0, then probs1 * scores1 = 0
    # If probs2 = 0, then scores2 -> inf. For KLD, probs1 * scores2 = 0 only if probs1 = 0 as well
    scores1 = pl.when(probs1 == 0).then(0).otherwise(scores1)
    scores2 = pl.when((probs1 == 0) & (probs2 == 0)).then(0).otherwise(scores2)

    kld = probs1 * scores2 - probs1 * scores1

    return kld
