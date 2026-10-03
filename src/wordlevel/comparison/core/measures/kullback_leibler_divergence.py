import polars as pl

from wordlevel.comparison.core.measures.entropy import tsallis_entropy_dual_dist_scores
from wordlevel.core.utils import alias_expressions, normalize_freqs


def kullback_leibler_divergence_dual_dist_scores(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    base: float = 2,
    normalize: bool = True,
    alias1: str | None = None,
    alias2: str | None = None,
) -> tuple[pl.Expr, pl.Expr]:
    """
    Per-token scores for the KL divergence shift D_KL(P_2 || P_1), expressed as a
    rearrangement of the paper's formula in surprisals s(p) = log_b(1/p) = -log_b p.

    Per-token contribution (arXiv:2008.02250 §III.C):

        delta = p_2 * s(p_1) - p_2 * s(p_2)
              = p_2 * log_b(1/p_1) - p_2 * log_b(1/p_2)
              = p_2 * log_b(p_2 / p_1)

    Mapping to the framework's `p_2 * s_2 - p_1 * s_1` with both component frequencies set
    to p_2 (via `share_component_freqs_by="comparison"`):
      - s_2 = s(p_1) = surprisal of freqs1
      - s_1 = s(p_2) = surprisal of freqs2

    So this function returns Tsallis surprisals at alpha=1 with their positions swapped
    relative to (freqs1, freqs2). Aligned with arXiv:2008.02250 §III.C and the standard
    info-theory convention (first argument of D is the data).
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
    """
    Clean swapped surprisals so the entropy convention 0 * s(0) = 0 holds, where s is the
    surprisal log_b(1/p). The two terms of the rearranged KL formula get asymmetric handling
    because only one of them should carry through a legitimate +inf divergence.

    `scores1` is s(p_2) (surprisal of freqs2 — the data); `scores2` is s(p_1) (surprisal
    of freqs1 — the codebook). Both will be multiplied by p_2 in the framework:

      - `p_2 * scores1` term: zero out scores1 when freqs2 == 0, since 0 * s(0) = 0 * inf
        would otherwise be NaN. This is the entropy convention 0 * log(0) = 0.
      - `p_2 * scores2` term: leave scores2 alone unless *both* freqs are zero.
          * p_2 > 0, p_1 = 0: scores2 = s(p_1) = +inf is preserved so the standalone
            metric produces a +inf per-token contribution — the divergence is legitimately
            infinite when the codebook lacks a token the data has.
          * p_2 = 0, p_1 > 0: scores2 is finite; 0 * finite = 0 naturally.
          * p_2 = 0, p_1 = 0: scores2 = +inf; 0 * inf = NaN — must clean.
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
    """
    KL divergence D_KL(P_2 || P_1) per token, expressed as a rearrangement in surprisals:

        D_KL = p_2 * s(p_1) - p_2 * s(p_2)
             = p_2 * log_b(1/p_1) - p_2 * log_b(1/p_2)
             = p_2 * log_b(p_2 / p_1)

    Let p_1 = freqs1 (the codebook / reference) and p_2 = freqs2 (the data / comparison).
    Aligned with arXiv:2008.02250 §III.C and standard info-theory convention.
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
