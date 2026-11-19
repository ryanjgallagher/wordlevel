import polars as pl

from wordlevel.utils import alias_expressions, normalize_freqs


def tsallis_entropy_dist_scores(
    freqs: pl.Expr,
    alpha: float = 1,
    base: float = 2,
    normalize: bool = True,
) -> pl.Expr:
    r"""
    Computes the Tsallis surprisal of each token, the score whose frequency-weighted average is the
    Tsallis entropy:

    $$
    s_\alpha(p_\tau) =
    \begin{cases}
        \dfrac{1 - p_\tau^{\alpha - 1}}{\alpha - 1} & \alpha \neq 1 \\
        \log_b \dfrac{1}{p_\tau} & \alpha = 1
    \end{cases}
    $$

    where $p_\tau$ is the normalized frequency of token $\tau$ and $b$ is the logarithm `base`,
    which is ignored unless $\alpha = 1$. If `normalize` is `False`, `freqs` are used as given
    rather than normalized to sum to 1.
    """
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
    r"""
    Sets the surprisal of each token with frequency 0 to 0, so that its term
    $p_\tau s_\alpha(p_\tau)$ in the entropy is 0 rather than undefined. This follows the
    convention $0 \log \frac{1}{0} = 0$.
    """
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
    """
    Computes the Tsallis surprisal of each token in each of the two corpora, named `alias1` and
    `alias2` if given.
    """
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
    """
    Sets the Tsallis surprisal of each token to 0 in each corpus where its frequency is 0, naming
    them `alias1` and `alias2` if given.
    """
    clean_scores1 = tsallis_entropy_clean_dist_scores(freqs=freqs1, scores=scores1)
    clean_scores2 = tsallis_entropy_clean_dist_scores(freqs=freqs2, scores=scores2)

    return alias_expressions(clean_scores1, clean_scores2, alias1, alias2)


def tsallis_entropy(
    freqs: pl.Expr,
    alpha: float = 1,
    base: float = 2,
    normalize: bool = True,
) -> pl.Expr:
    """
    Not exposed in the API. Computes each token's contribution to the Tsallis entropy of a corpus,
    which sum to the entropy.
    """
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
    """
    Not exposed in the API. Computes each token's contribution to the Shannon entropy of a corpus,
    which sum to the entropy.
    """
    return tsallis_entropy(freqs, alpha=1, base=base, normalize=normalize)
