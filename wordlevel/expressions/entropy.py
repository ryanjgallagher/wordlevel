import polars as pl

from wordlevel.expressions.utils import normalize_freqs, alias_expressions


def tsallis_entropy_weighted_avg_scores(
    freqs: pl.Expr,
    alpha: float = 1,
    base: float = 2,
    normalize: bool = True,
) -> pl.Expr:
    """ """
    probs = normalize_freqs(freqs, normalize=normalize)

    if alpha == 1:
        scores = -1 * probs.log(base=base)
    else:
        scores = (1 / (alpha - 1)) * (1 - probs.pow(alpha - 1))

    return scores


def tsallis_entropy_shift_scores(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    alpha: float = 1,
    base: float = 2,
    normalize: bool = True,
    alias1: str | None = None,
    alias2: str | None = None,
) -> tuple[pl.Expr, pl.Expr]:
    """ """
    scores1 = tsallis_entropy_weighted_avg_scores(
        freqs1, alpha=alpha, base=base, normalize=normalize
    )
    scores2 = tsallis_entropy_weighted_avg_scores(
        freqs2, alpha=alpha, base=base, normalize=normalize
    )

    return alias_expressions(scores1, scores2, alias1, alias2)


def tsallis_entropy(
    freqs: pl.Expr,
    alpha: float = 1,
    base: float = 2,
    normalize: bool = True,
) -> pl.Expr:
    """ """
    probs = normalize_freqs(freqs, normalize=normalize)

    scores = (
        pl.when(probs > 0)
        .then(tsallis_entropy_weighted_avg_scores(probs, alpha=alpha, base=base, normalize=False))
        .otherwise(0)
    )
    entropies = probs * scores

    return entropies


def shannon_entropy(
    freqs: pl.Expr,
    base: float = 2,
    normalize: bool = True,
) -> pl.Expr:
    """ """
    return tsallis_entropy(freqs, alpha=1, base=base, normalize=normalize)
