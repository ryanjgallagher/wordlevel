import polars as pl

from wordlevel.utils import alias_expressions, normalize_freqs


def tsallis_entropy_dist_scores(
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


def tsallis_entropy_clean_dist_scores(
    freqs: pl.Expr,
    scores: pl.Expr,
) -> pl.Expr:
    """ """
    return pl.when(freqs > 0).then(scores).otherwise(0)


def tsallis_entropy_dual_dist_scores(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    alpha: float = 1,
    base: float = 2,
    normalize: bool = True,
    alias1: str | None = None,
    alias2: str | None = None,
) -> tuple[pl.Expr, pl.Expr]:
    """ """
    scores1 = tsallis_entropy_dist_scores(freqs1, alpha=alpha, base=base, normalize=normalize)
    scores2 = tsallis_entropy_dist_scores(freqs2, alpha=alpha, base=base, normalize=normalize)

    return alias_expressions(scores1, scores2, alias1, alias2)


def tsallis_entropy_clean_dual_dist_scores(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    scores1: pl.Expr,
    scores2: pl.Expr,
    alias1: str | None = None,
    alias2: str | None = None,
) -> tuple[pl.Expr, pl.Expr]:
    """ """
    clean_scores1 = tsallis_entropy_clean_dist_scores(freqs=freqs1, scores=scores1)
    clean_scores2 = tsallis_entropy_clean_dist_scores(freqs=freqs2, scores=scores2)

    return alias_expressions(clean_scores1, clean_scores2, alias1, alias2)


def tsallis_entropy(
    freqs: pl.Expr,
    alpha: float = 1,
    base: float = 2,
    normalize: bool = True,
) -> pl.Expr:
    """ """
    probs = normalize_freqs(freqs, normalize=normalize)
    scores = tsallis_entropy_dist_scores(probs, alpha=alpha, base=base, normalize=False)
    clean_scores = tsallis_entropy_clean_dist_scores(freqs=probs, scores=scores)

    entropies = probs * clean_scores

    return entropies


def shannon_entropy(
    freqs: pl.Expr,
    base: float = 2,
    normalize: bool = True,
) -> pl.Expr:
    """ """
    return tsallis_entropy(freqs, alpha=1, base=base, normalize=normalize)
