import polars as pl

from wordlevel.expressions.utils import normalize_freqs, mixed_distribution, alias_expressions


def jensen_shannon_divergence_shift_scores(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    alpha: float = 1,
    base: float = 2,
    weight1: float = 0.5,
    weight2: float = 0.5,
    normalize: bool = True,
    alias1: str | None = None,
    alias2: str | None = None,
) -> tuple[pl.Expr, pl.Expr]:
    """ """
    probs1 = normalize_freqs(freqs1, normalize=normalize)
    probs2 = normalize_freqs(freqs2, normalize=normalize)

    if weight1 + weight2 != 1:
        weight1, weight2 = weight1 / (weight1 + weight2), weight2 / (weight1 + weight2)

    probs_mixed = mixed_distribution(
        probs1, probs2, weight1=weight1, weight2=weight2, normalize=False
    )

    if alpha == 1:
        scores1 = weight1 * (probs_mixed.log(base=base) - probs1.log(base=base))
        scores2 = weight2 * (probs2.log(base=base) - probs_mixed.log(base=base))
    else:
        scores1 = weight1 * (probs_mixed.pow(alpha - 1) - probs1.pow(alpha - 1)) / (alpha - 1)
        scores2 = weight2 * (probs2.pow(alpha - 1) - probs_mixed.pow(alpha - 1)) / (alpha - 1)

    return alias_expressions(scores1, scores2, alias1, alias2)


def jensen_shannon_divergence(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    weight1: float = 0.5,
    weight2: float = 0.5,
    base: int = 2,
    alpha: float = 1,
    normalize: bool = True,
) -> pl.Expr:
    """ """
    probs1 = normalize_freqs(freqs1, normalize=normalize)
    probs2 = normalize_freqs(freqs2, normalize=normalize)

    scores1, scores2 = jensen_shannon_divergence_shift_scores(
        probs1, probs2, alpha=alpha, base=base, weight1=weight1, weight2=weight2, normalize=False
    )
    scores1 = pl.when(probs1 > 0).then(scores1).otherwise(0)
    scores2 = pl.when(probs2 > 0).then(scores2).otherwise(0)

    jsd = probs2 * scores2 - probs1 * scores1

    return jsd
