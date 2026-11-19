import polars as pl

from wordlevel.comparison.core.measures.entropy import tsallis_entropy_dual_dist_scores
from wordlevel.utils import alias_expressions, normalize_freqs


def kullback_leibler_divergence_dual_dist_scores(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    base: float = 2,
    normalize: bool = True,
    alias1: str | None = None,
    alias2: str | None = None,
) -> tuple[pl.Expr, pl.Expr]:
    r"""
    Computes each token's score in each of the two corpora for the Kullback-Leibler divergence
    $D_{\text{KL}} \left( P^{(2)} \parallel P^{(1)} \right)$ of the second corpus from the first.
    The scores are named `alias1` and `alias2` if given.

    The scores are

    $$
    \phi_\tau^{(1)} = \log_b \frac{1}{p_\tau^{(2)}}
    \qquad
    \phi_\tau^{(2)} = \log_b \frac{1}{p_\tau^{(1)}}
    $$

    where $p_\tau^{(1)}$ and $p_\tau^{(2)}$ are the normalized frequencies of token $\tau$ in each
    corpus and $b$ is the logarithm `base`. Each corpus is scored by the other corpus's
    frequencies, so that when both scores are weighted by the frequency in the second corpus, each
    token's contribution to $D_{\text{KL}} \left( P^{(2)} \parallel P^{(1)} \right)$ is

    $$
    p_\tau^{(2)} \phi_\tau^{(2)} - p_\tau^{(2)} \phi_\tau^{(1)}
        = p_\tau^{(2)} \log_b \frac{1}{p_\tau^{(1)}} - p_\tau^{(2)} \log_b \frac{1}{p_\tau^{(2)}}
        = p_\tau^{(2)} \log_b \frac{p_\tau^{(2)}}{p_\tau^{(1)}}
    $$

    If `normalize` is `False`, `freqs1` and `freqs2` are used as given rather than normalized to
    sum to 1.
    """
    surprisal_freqs1, surprisal_freqs2 = tsallis_entropy_dual_dist_scores(
        freqs1, freqs2, alpha=1, base=base, normalize=normalize
    )
    return alias_expressions(surprisal_freqs2, surprisal_freqs1, alias1, alias2)


def kullback_leibler_divergence_clean_dual_dist_scores(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    scores1: pl.Expr,
    scores2: pl.Expr,
    alias1: str | None = None,
    alias2: str | None = None,
) -> tuple[pl.Expr, pl.Expr]:
    r"""
    Sets each token's scores to 0 where they are infinite and the second corpus does not use it, so
    that its terms $p_\tau^{(2)} \phi_\tau^{(1)}$ and $p_\tau^{(2)} \phi_\tau^{(2)}$ in the
    Kullback-Leibler divergence $D_{\text{KL}} \left( P^{(2)} \parallel P^{(1)} \right)$ are 0
    rather than undefined. The scores are named `alias1` and `alias2` if given.

    The scores are $\phi_\tau^{(1)} = \log_b \frac{1}{p_\tau^{(2)}}$ and
    $\phi_\tau^{(2)} = \log_b \frac{1}{p_\tau^{(1)}}$, and are set to 0 asymmetrically:

    - $\phi_\tau^{(1)}$ is set to 0 wherever $p_\tau^{(2)} = 0$, where it is infinite. This follows
      the convention $0 \log \frac{1}{0} = 0$.
    - $\phi_\tau^{(2)}$ is set to 0 only where $p_\tau^{(1)} = p_\tau^{(2)} = 0$. Where only
      $p_\tau^{(2)} = 0$, it is finite, so its term is 0 without being set. Where only
      $p_\tau^{(1)} = 0$, it stays infinite, since the divergence is infinite when the first corpus
      does not use a token that the second corpus does.
    """
    clean_scores1 = pl.when(freqs2 == 0).then(0).otherwise(scores1)
    clean_scores2 = pl.when((freqs2 == 0) & (freqs1 == 0)).then(0).otherwise(scores2)

    return alias_expressions(clean_scores1, clean_scores2, alias1, alias2)


def kullback_leibler_divergence(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    base: int = 2,
    normalize: bool = True,
) -> pl.Expr:
    r"""
    Not exposed in the API. Computes each token's contribution to the Kullback-Leibler divergence
    $D_{\text{KL}} \left( P^{(2)} \parallel P^{(1)} \right)$ of the second corpus from the first,
    which sum to the divergence.
    """
    probs1 = normalize_freqs(freqs1, normalize=normalize)
    probs2 = normalize_freqs(freqs2, normalize=normalize)

    scores1, scores2 = kullback_leibler_divergence_dual_dist_scores(
        probs1, probs2, base=base, normalize=False
    )
    # After the swap inside the dual_dist helper: scores1 = s(p_2), scores2 = s(p_1).
    # The per-token KL is p_2 * s(p_1) - p_2 * s(p_2) = probs2 * scores2 - probs2 * scores1.
    clean_scores1, clean_scores2 = kullback_leibler_divergence_clean_dual_dist_scores(
        freqs1=probs1, freqs2=probs2, scores1=scores1, scores2=scores2
    )

    kld = probs2 * clean_scores2 - probs2 * clean_scores1

    return kld
