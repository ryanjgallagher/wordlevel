import polars as pl

from wordlevel.utils import alias_expressions, mixed_distribution, normalize_freqs


def jensen_shannon_divergence_dual_dist_scores(
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
    r"""
    Computes each token's score in each of the two corpora for the Jensen-Shannon divergence, named
    `alias1` and `alias2` if given.

    The divergence compares each corpus to their mixture, in which a token $\tau$ has the
    normalized frequency $p_\tau^{(M)} = \pi_1 \, p_\tau^{(1)} + \pi_2 \, p_\tau^{(2)}$, where
    $\pi_1$ and $\pi_2$ are `weight1` and `weight2`. If the weights do not sum to 1, they are
    rescaled so that they do. The scores are

    $$
    \phi_\tau^{(1)} =
    \begin{cases}
        \pi_1 \dfrac{\left[ p_\tau^{(M)} \right]^{\alpha - 1}
            - \left[ p_\tau^{(1)} \right]^{\alpha - 1}}{\alpha - 1} & \alpha \neq 1 \\
        \pi_1 \log_b \dfrac{p_\tau^{(M)}}{p_\tau^{(1)}} & \alpha = 1
    \end{cases}
    \qquad
    \phi_\tau^{(2)} =
    \begin{cases}
        \pi_2 \dfrac{\left[ p_\tau^{(2)} \right]^{\alpha - 1}
            - \left[ p_\tau^{(M)} \right]^{\alpha - 1}}{\alpha - 1} & \alpha \neq 1 \\
        \pi_2 \log_b \dfrac{p_\tau^{(2)}}{p_\tau^{(M)}} & \alpha = 1
    \end{cases}
    $$

    so that each token's contribution to the divergence is
    $p_\tau^{(2)} \phi_\tau^{(2)} - p_\tau^{(1)} \phi_\tau^{(1)}$. Here, $b$ is the logarithm
    `base`, which is ignored unless $\alpha = 1$. If `normalize` is `False`, `freqs1` and `freqs2`
    are used as given rather than normalized to sum to 1.
    """
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


def jensen_shannon_divergence_clean_dual_dist_scores(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    scores1: pl.Expr,
    scores2: pl.Expr,
    alias1: str | None = None,
    alias2: str | None = None,
) -> tuple[pl.Expr, pl.Expr]:
    r"""
    Sets each token's score to 0 in each corpus where its frequency is 0, so that its terms
    $p_\tau^{(1)} \phi_\tau^{(1)}$ and $p_\tau^{(2)} \phi_\tau^{(2)}$ in the divergence are 0 rather
    than undefined. The scores are named `alias1` and `alias2` if given.
    """
    clean_scores1 = pl.when(freqs1 > 0).then(scores1).otherwise(0)
    clean_scores2 = pl.when(freqs2 > 0).then(scores2).otherwise(0)

    return alias_expressions(clean_scores1, clean_scores2, alias1, alias2)


def jensen_shannon_divergence(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    weight1: float = 0.5,
    weight2: float = 0.5,
    base: int = 2,
    alpha: float = 1,
    normalize: bool = True,
) -> pl.Expr:
    """
    Not exposed in the API. Computes each token's contribution to the Jensen-Shannon divergence
    between two corpora, which sum to the divergence.
    """
    probs1 = normalize_freqs(freqs1, normalize=normalize)
    probs2 = normalize_freqs(freqs2, normalize=normalize)

    scores1, scores2 = jensen_shannon_divergence_dual_dist_scores(
        probs1, probs2, alpha=alpha, base=base, weight1=weight1, weight2=weight2, normalize=False
    )
    clean_scores1, clean_scores2 = jensen_shannon_divergence_clean_dual_dist_scores(
        freqs1=probs1, freqs2=probs2, scores1=scores1, scores2=scores2
    )

    jsd = probs2 * clean_scores2 - probs1 * clean_scores1

    return jsd
