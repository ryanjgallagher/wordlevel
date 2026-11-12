from math import isinf

import polars as pl


def rank(expr: pl.Expr, alpha: float = 1) -> pl.Expr:
    """
    Ranks in descending order of frequency, with ties taking the mean of the ranks they span,
    raised to the power alpha (except at alpha=0 and alpha=inf, whose limits use the raw ranks).

    Raises ValueError if alpha is negative.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    ranks = expr.rank(method="average", descending=True)

    if (alpha > 0) and (not isinf(alpha)):
        ranks = ranks.pow(alpha)

    return ranks


def rank_divergence(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    alpha: float = 1.0,
) -> pl.Expr:
    """
    Per-token rank-turbulence divergence. The contributions sum to the divergence, which is 0
    for identical rankings and 1 for disjoint corpora.

    Reference:
        > Dodds, P. S., Minot, J. R., Arnold, M. V., Alshaabi, T., Adams, J. L.,
        Dewhurst, D. R., Gray, T. J., Frank, M. R., Reagan, A. J., & Danforth, C. M. (2023).
        [Allotaxonometry and rank-turbulence divergence: A universal instrument for comparing
        complex systems](https://link.springer.com/article/10.1140/epjds/s13688-023-00400-x).
        EPJ Data Science, 12(1), 37.
    """
    summands = rank_divergence_summands(freqs1, freqs2, alpha=alpha)
    norm = rank_divergence_norm_summands(freqs1, freqs2, alpha=alpha).sum()

    return summands / norm


def rank_divergence_summands(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    alpha: float = 1.0,
) -> pl.Expr:
    r"""
    Unnormalized per-token rank-turbulence divergence:

    $$
    \delta_\tau =
    \begin{cases}
        \dfrac{\alpha + 1}{\alpha}
        \left|
            \dfrac{1}{\left[ r_\tau^{(1)} \right]^\alpha}
            - \dfrac{1}{\left[ r_\tau^{(2)} \right]^\alpha}
        \right|^{1 / (\alpha + 1)}
            & 0 < \alpha < \infty \\
        \left| \ln \left( r_\tau^{(1)} / r_\tau^{(2)} \right) \right|
            & \alpha = 0 \\
        \mathbb{1}\left[ r_\tau^{(1)} \neq r_\tau^{(2)} \right]
        \max \left\lbrace 1 / r_\tau^{(1)}, 1 / r_\tau^{(2)} \right\rbrace
            & \alpha = \infty
    \end{cases}
    $$

    where $r_\tau^{(1)}$ and $r_\tau^{(2)}$ are the ranks of token $\tau$ in each corpus.
    """
    ranks1 = rank(freqs1, alpha=alpha)
    ranks2 = rank(freqs2, alpha=alpha)

    if alpha == 0:
        summands = (ranks1.log() - ranks2.log()).abs()
    elif isinf(alpha):
        summands = (
            pl.when(ranks1 == ranks2).then(0).otherwise(pl.max_horizontal(1 / ranks1, 1 / ranks2))
        )
    else:
        prefactor = (alpha + 1.0) / alpha
        summands = prefactor * (1 / ranks1 - 1 / ranks2).abs().pow(1.0 / (alpha + 1))

    return summands


def rank_divergence_norm_summands(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    alpha: float = 1.0,
) -> pl.Expr:
    r"""
    Per-token terms of the rank-turbulence divergence's normalization $\mathcal{N}_\alpha$:
    the unnormalized divergence the two corpora would have if they shared no tokens but kept
    their rankings, so normalizing by it gives disjoint corpora a divergence of 1.

    $$
    \mathcal{N}_\alpha =
    \frac{\alpha + 1}{\alpha}
    \sum_{\tau \in 1}
    \left|
        \frac{1}{\left[ r_\tau^{(1)} \right]^\alpha}
        - \frac{1}{\left[ b^{(2)} \right]^\alpha}
    \right|^{1 / (\alpha + 1)}
    + \frac{\alpha + 1}{\alpha}
    \sum_{\tau \in 2}
    \left|
        \frac{1}{\left[ r_\tau^{(2)} \right]^\alpha}
        - \frac{1}{\left[ b^{(1)} \right]^\alpha}
    \right|^{1 / (\alpha + 1)}
    $$

    with the limits

    $$
    \mathcal{N}_0 =
    \sum_{\tau \in 1} \left| \ln \left( r_\tau^{(1)} / b^{(2)} \right) \right|
    + \sum_{\tau \in 2} \left| \ln \left( r_\tau^{(2)} / b^{(1)} \right) \right|
    $$

    $$
    \mathcal{N}_\infty =
    \sum_{\tau \in 1}
    \mathbb{1}\left[ r_\tau^{(1)} \neq b^{(2)} \right]
    \max \left\lbrace 1 / r_\tau^{(1)}, 1 / b^{(2)} \right\rbrace
    + \sum_{\tau \in 2}
    \mathbb{1}\left[ r_\tau^{(2)} \neq b^{(1)} \right]
    \max \left\lbrace 1 / r_\tau^{(2)}, 1 / b^{(1)} \right\rbrace
    $$

    where $\tau \in 1$ and $\tau \in 2$ range over the tokens with nonzero frequency in each
    corpus, and $r_\tau^{(1)}$ and $r_\tau^{(2)}$ are the token's ranks. Without shared tokens,
    each corpus's tokens would tie for last place in the other corpus, at the mean of the
    positions after that corpus's own tokens:

    $$
    b^{(2)} = N_2 + \frac{N_1 + 1}{2},
    \qquad
    b^{(1)} = N_1 + \frac{N_2 + 1}{2}
    $$

    where $N_1$ and $N_2$ are the numbers of tokens with nonzero frequency in each corpus.
    Each token's term is its summand from the first sum plus its summand from the second.

    This differs from the paper's normalization, which incorrectly places the tie at
    $N_2 + N_1 / 2$ rather than the mean of the positions, pairs each corpus's ranks with its
    own tied last rank rather than the other corpus's, and at $\alpha = \infty$ leaves out the
    tied last ranks. Our fixes of these issues ensure that disjoint corpora of different sizes
    have a divergence of 1.
    """
    n_types1 = freqs1.filter(freqs1 > 0).len()
    n_types2 = freqs2.filter(freqs2 > 0).len()

    comp1 = rank_divergence_norm_summand_component(freqs1, n_types2, n_types1, alpha)
    comp2 = rank_divergence_norm_summand_component(freqs2, n_types1, n_types2, alpha)

    return comp1 + comp2


def rank_divergence_norm_summand_component(
    freqs: pl.Expr,
    n_types1: pl.Expr,
    n_types2: pl.Expr,
    alpha: float = 1.0,
) -> pl.Expr:
    r"""
    One corpus's per-token terms of the rank-turbulence divergence's normalization: the
    unnormalized divergence between each token's rank $r_\tau$ and the rank $b$ that the
    corpus's tokens would tie at in the other corpus if the two shared no tokens.

    $$
    \nu_\tau =
    \begin{cases}
        \dfrac{\alpha + 1}{\alpha}
        \left|
            \dfrac{1}{r_\tau^\alpha} - \dfrac{1}{b^\alpha}
        \right|^{1 / (\alpha + 1)}
            & 0 < \alpha < \infty \\
        \left| \ln \left( r_\tau / b \right) \right|
            & \alpha = 0 \\
        \mathbb{1}\left[ r_\tau \neq b \right]
        \max \left\lbrace 1 / r_\tau, 1 / b \right\rbrace
            & \alpha = \infty
    \end{cases}
    \qquad
    b = n_1 + \frac{n_2 + 1}{2}
    $$

    where $n_1$ (`n_types1`) is the number of tokens with nonzero frequency in the other
    corpus and $n_2$ (`n_types2`) is the number in this corpus. $b$ is the mean of the positions
    the tie spans, matching how tied ranks are assigned. Tokens with frequency 0 contribute 0.
    """
    ranks = rank(freqs, alpha=alpha)
    tied_last_rank = n_types1 + 0.5 * n_types2 + 0.5

    if alpha == 0:
        summand = (ranks.log() - tied_last_rank.log()).abs()
    elif isinf(alpha):
        summand = (
            pl.when(ranks == tied_last_rank)
            .then(0)
            .otherwise(pl.max_horizontal(1 / ranks, 1 / tied_last_rank))
        )
    else:
        # fmt: off
        summand = (
            ((alpha + 1.0) / alpha)
            * (1 / ranks - 1 / tied_last_rank ** alpha)
            .abs()
            .pow(1.0 / (alpha + 1))
        )
        # fmt: on

    component = pl.when(freqs > 0).then(summand).otherwise(0)

    return component
