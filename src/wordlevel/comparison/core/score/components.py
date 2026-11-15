from typing import Literal

import polars as pl
from polars import col

from wordlevel.utils import are_same_sign, normalize_freqs


# --------------------------------------------------------------------
# Components
# --------------------------------------------------------------------


def get_shift_components(
    df: pl.DataFrame | pl.LazyFrame,
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    scores1: pl.Expr,
    scores2: pl.Expr,
    reference_score: float | Literal["mean"] | pl.Expr,
    normalize: bool = True,
    include_total_difference: bool = True,
    include_rank: bool = False,
    frequency_diff_alias: str = "freq_diff",
    weighted_score_diff_alias: str = "weighted_score_diff",
    reference_score_diff_alias: str = "ref_score_diff",
    total_difference_alias: str = "total_difference",
    rank_alias: str = "rank",
) -> pl.DataFrame | pl.LazyFrame:
    r"""
    Appends columns holding the components of each token's contribution to a difference in weighted
    averages, where each corpus may assign a token its own score.

    The contribution $\delta_\tau$ of a token $\tau$ decomposes into three components:

    $$
    \delta_\tau =
        \underbrace{
            \left( p_\tau^{(2)} - p_\tau^{(1)} \right)
        }_{\text{frequency difference}}
        \underbrace{
            \left[
                \frac{1}{2} \left( \phi_\tau^{(1)} + \phi_\tau^{(2)} \right) - \Phi^{(\text{ref})}
            \right]
        }_{\text{reference score difference}}
        +
        \underbrace{
            \frac{1}{2} \left( p_\tau^{(1)} + p_\tau^{(2)} \right)
            \left( \phi_\tau^{(2)} - \phi_\tau^{(1)} \right)
        }_{\text{weighted score difference}}
    $$

    where $p_\tau^{(1)}$ and $p_\tau^{(2)}$ are the normalized frequencies of the token in each
    corpus, $\phi_\tau^{(1)}$ and $\phi_\tau^{(2)}$ are its scores, and $\Phi^{(\text{ref})}$ is
    the reference score. When the frequencies are normalized, the contributions sum to the
    difference in weighted averages
    $\sum_\tau p_\tau^{(2)} \phi_\tau^{(2)} - \sum_\tau p_\tau^{(1)} \phi_\tau^{(1)}$

    Reference:
        > Gallagher, R. J., Frank, M. R., Mitchell, L., Schwartz, A. J., Reagan, A. J.,
        Danforth, C. M., & Dodds, P. S. (2021). [Generalized word shift graphs: a method for
        visualizing and explaining pairwise comparisons
        between texts](https://link.springer.com/article/10.1140/epjds/s13688-021-00260-3).
        EPJ Data Science, 10(1), 4.

    Args:
        df:
            The dataframe to append the component columns to
        freqs1:
            Frequency of each token in the first corpus (the reference)
        freqs2:
            Frequency of each token in the second corpus (the comparison)
        scores1:
            Score of each token in the first corpus, $\phi_\tau^{(1)}$
        scores2:
            Score of each token in the second corpus, $\phi_\tau^{(2)}$
        reference_score:
            The reference score $\Phi^{(\text{ref})}$. If `"mean"`, uses the weighted average of
            the first corpus (the reference corpus), $\sum_\tau p_\tau^{(1)} \phi_\tau^{(1)}$
        normalize:
            Whether to normalize `freqs1` and `freqs2` to sum to 1. If `False`, they are used as
            given
        include_total_difference:
            Whether to add a column holding each token's contribution $\delta_\tau$
        include_rank:
            Whether to add a column ranking tokens by the magnitude of their contribution, where
            rank 1 is the largest and ties are broken by row order. Ignored unless
            `include_total_difference`
        frequency_diff_alias:
            Name of the column holding the frequency difference
        weighted_score_diff_alias:
            Name of the column holding the weighted score difference
        reference_score_diff_alias:
            Name of the column holding the reference score difference
        total_difference_alias:
            Name of the column holding the contribution $\delta_\tau$
        rank_alias:
            Name of the column holding the rank

    Returns:
        `df` with the component columns added
    """
    if normalize:
        freqs1 = normalize_freqs(freqs1)
        freqs2 = normalize_freqs(freqs2)

    # fmt: off
    df_comps = df.with_columns(
        frequency_diff(freqs1=freqs1, freqs2=freqs2, normalize=False).alias(frequency_diff_alias),
        weighted_score_diff(
            freqs1=freqs1, freqs2=freqs2, scores1=scores1, scores2=scores2, normalize=False
        ).alias(weighted_score_diff_alias),
        reference_score_diff(
            scores1=scores1, scores2=scores2, reference_score=reference_score, freqs1=freqs1, normalize=False
        ).alias(reference_score_diff_alias)
    )

    if include_total_difference:
        df_comps = df_comps.with_columns(
            (col(frequency_diff_alias) * col(reference_score_diff_alias) + col(weighted_score_diff_alias))
            .alias(total_difference_alias),
        )

        if include_rank:
            df_comps = df_comps.with_columns(
                col(total_difference_alias).abs().rank(descending=True, method='ordinal').alias(rank_alias),
            )
    # fmt: on

    return df_comps


def frequency_diff(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    normalize: bool = True,
) -> pl.Expr:
    r"""
    Computes the difference in each token's normalized frequency between the two corpora:

    $$
    p_\tau^{(2)} - p_\tau^{(1)}
    $$

    If `normalize` is `False`, `freqs1` and `freqs2` are used as given rather than normalized to sum
    to 1.
    """
    probs1 = normalize_freqs(freqs1, normalize=normalize)
    probs2 = normalize_freqs(freqs2, normalize=normalize)

    return probs2 - probs1


def weighted_score_diff(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    scores1: pl.Expr,
    scores2: pl.Expr,
    normalize: bool = True,
) -> pl.Expr:
    r"""
    Computes the difference in each token's score between the two corpora, weighted by its average
    normalized frequency:

    $$
    \frac{1}{2} \left( p_\tau^{(1)} + p_\tau^{(2)} \right)
    \left( \phi_\tau^{(2)} - \phi_\tau^{(1)} \right)
    $$

    If `normalize` is `False`, `freqs1` and `freqs2` are used as given rather than normalized to sum
    to 1.
    """
    probs1 = normalize_freqs(freqs1, normalize=normalize)
    probs2 = normalize_freqs(freqs2, normalize=normalize)

    return (0.5 * probs1 + 0.5 * probs2) * (scores2 - scores1)


def reference_score_diff(
    scores1: pl.Expr,
    scores2: pl.Expr,
    reference_score: float | Literal["mean"] | pl.Expr,
    freqs1: pl.Expr | None = None,
    normalize: bool = True,
) -> pl.Expr:
    r"""
    Computes the difference between each token's average score across the two corpora and the
    reference score:

    $$
    \frac{1}{2} \left( \phi_\tau^{(1)} + \phi_\tau^{(2)} \right) - \Phi^{(\text{ref})}
    $$

    A positive difference means the token is relatively positive. If `reference_score` is `"mean"`,
    the reference score is the weighted average of the first corpus,
    $\Phi^{(\text{ref})} = \sum_\tau p_\tau^{(1)} \phi_\tau^{(1)}$, whose frequencies are
    normalized to sum to 1 unless `normalize` is `False`.

    Raises:
        ValueError: If `reference_score` is `"mean"` and `freqs1` is not given.
    """
    if isinstance(reference_score, str) and reference_score == "mean":
        if freqs1 is None:
            raise ValueError('Must pass `freqs1` if `reference_score` is "mean"')

        reference_score = (scores1 * normalize_freqs(freqs1, normalize=normalize)).sum()

    return (0.5 * scores1 + 0.5 * scores2) - reference_score


# --------------------------------------------------------------------
# Plot components
# --------------------------------------------------------------------


def get_shift_plot_components(
    df: pl.DataFrame | pl.LazyFrame,
    frequency_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    more_of_positive_alias: str = "more_of_positive",
    less_of_negative_alias: str = "less_of_negative",
    more_of_negative_alias: str = "more_of_negative",
    less_of_positive_alias: str = "less_of_positive",
    higher_score_alias: str = "higher_score",
    lower_score_alias: str = "lower_score",
    counteracted_suffix: str = "_counteracted",
) -> pl.DataFrame | pl.LazyFrame:
    r"""
    Appends columns that split each token's contribution into the bars of a word shift graph.

    A token's contribution is the sum of a frequency component and a score component:

    $$
    \delta_\tau =
        \underbrace{
            \left( p_\tau^{(2)} - p_\tau^{(1)} \right)
            \left[
                \frac{1}{2} \left( \phi_\tau^{(1)} + \phi_\tau^{(2)} \right) - \Phi^{(\text{ref})}
            \right]
        }_{\text{frequency component}}
        +
        \underbrace{
            \frac{1}{2} \left( p_\tau^{(1)} + p_\tau^{(2)} \right)
            \left( \phi_\tau^{(2)} - \phi_\tau^{(1)} \right)
        }_{\text{score component}}
    $$

    Each component is assigned a bar type by its signs:

    | Bar type | Component | Condition |
    |----------|-----------|-----------|
    | `more_of_positive` $(+\uparrow)$ | Frequency | $p_\tau^{(2)} \ge p_\tau^{(1)}$ and $\frac{1}{2} \left( \phi_\tau^{(1)} + \phi_\tau^{(2)} \right) \ge \Phi^{(\text{ref})}$ |
    | `less_of_positive` $(+\downarrow)$ | Frequency | $p_\tau^{(2)} < p_\tau^{(1)}$ and $\frac{1}{2} \left( \phi_\tau^{(1)} + \phi_\tau^{(2)} \right) \ge \Phi^{(\text{ref})}$ |
    | `more_of_negative` $(-\uparrow)$ | Frequency | $p_\tau^{(2)} \ge p_\tau^{(1)}$ and $\frac{1}{2} \left( \phi_\tau^{(1)} + \phi_\tau^{(2)} \right) < \Phi^{(\text{ref})}$ |
    | `less_of_negative` $(-\downarrow)$ | Frequency | $p_\tau^{(2)} < p_\tau^{(1)}$ and $\frac{1}{2} \left( \phi_\tau^{(1)} + \phi_\tau^{(2)} \right) < \Phi^{(\text{ref})}$ |
    | `higher_score` $(\bigtriangleup)$ | Score | $\phi_\tau^{(2)} \ge \phi_\tau^{(1)}$ |
    | `lower_score` $(\bigtriangledown)$ | Score | $\phi_\tau^{(2)} < \phi_\tau^{(1)}$ |

    When the two components have opposite signs they partly cancel, so each bar type also has a
    counteracted column holding the part of its component that was cancelled:

    - If the components share a sign, each bar is its whole component, and the counteracted bars
      are 0. A component of 0 shares a sign with any value.
    - If the components have opposite signs, the component that is larger in magnitude has a bar
      holding the sum of the two components, and a counteracted bar holding the part the smaller
      component cancelled. A component that is not larger in magnitude has a bar of 0, and a
      counteracted bar holding the whole component.

    So the bars of a token sum to its contribution $\delta_\tau$, and its two counteracted bars
    cancel each other out.

    Reference:
        > Gallagher, R. J., Frank, M. R., Mitchell, L., Schwartz, A. J., Reagan, A. J.,
        Danforth, C. M., & Dodds, P. S. (2021). [Generalized word shift graphs: a method for
        visualizing and explaining pairwise comparisons
        between texts](https://link.springer.com/article/10.1140/epjds/s13688-021-00260-3).
        EPJ Data Science, 10(1), 4.

    Args:
        df:
            The dataframe to append the bar columns to
        frequency_diff:
            Difference in each token's normalized frequency, $p_\tau^{(2)} - p_\tau^{(1)}$
        weighted_score_diff:
            The score component, $\frac{1}{2} \left( p_\tau^{(1)} + p_\tau^{(2)} \right)
            \left( \phi_\tau^{(2)} - \phi_\tau^{(1)} \right)$
        reference_score_diff:
            Difference between each token's average score and the reference score,
            $\frac{1}{2} \left( \phi_\tau^{(1)} + \phi_\tau^{(2)} \right) - \Phi^{(\text{ref})}$
        more_of_positive_alias:
            Name of the column holding the `more_of_positive` bar
        less_of_negative_alias:
            Name of the column holding the `less_of_negative` bar
        more_of_negative_alias:
            Name of the column holding the `more_of_negative` bar
        less_of_positive_alias:
            Name of the column holding the `less_of_positive` bar
        higher_score_alias:
            Name of the column holding the `higher_score` bar
        lower_score_alias:
            Name of the column holding the `lower_score` bar
        counteracted_suffix:
            Suffix appended to each bar's column name to name its counteracted column

    Returns:
        `df` with the twelve bar columns appended
    """
    components = {
        "frequency_diff": frequency_diff,
        "weighted_score_diff": weighted_score_diff,
        "reference_score_diff": reference_score_diff,
    }

    # fmt: off
    df_comps = df.with_columns(
        more_of_positive(**components, counteracted=False).alias(more_of_positive_alias),
        less_of_negative(**components, counteracted=False).alias(less_of_negative_alias),
        more_of_negative(**components, counteracted=False).alias(more_of_negative_alias),
        less_of_positive(**components, counteracted=False).alias(less_of_positive_alias),
        higher_score(**components, counteracted=False).alias(higher_score_alias),
        lower_score(**components, counteracted=False).alias(lower_score_alias),
        more_of_positive(**components, counteracted=True).alias(f"{more_of_positive_alias}{counteracted_suffix}"),
        less_of_negative(**components, counteracted=True).alias(f"{less_of_negative_alias}{counteracted_suffix}"),
        more_of_negative(**components, counteracted=True).alias(f"{more_of_negative_alias}{counteracted_suffix}"),
        less_of_positive(**components, counteracted=True).alias(f"{less_of_positive_alias}{counteracted_suffix}"),
        higher_score(**components, counteracted=True).alias(f"{higher_score_alias}{counteracted_suffix}"),
        lower_score(**components, counteracted=True).alias(f"{lower_score_alias}{counteracted_suffix}"),
    )
    # fmt: on

    return df_comps


def plot_component(component1: pl.Expr, component2: pl.Expr) -> pl.Expr:
    r"""
    Computes what remains of `component1` after `component2`, the other component of the same
    contribution, cancels against it.

    With $a$ as `component1` and $b$ as `component2`:

    $$
    \begin{cases}
        a & a \text{ and } b \text{ share a sign} \\
        a + b & |a| > |b| \\
        0 & \text{otherwise}
    \end{cases}
    $$

    - If $a$ and $b$ share a sign, both push the contribution in the same direction, so nothing
      cancels and all of $a$ remains.
    - If they have opposite signs and $a$ is larger in magnitude, $b$ cancels part of $a$. What
      remains of $a$ is $a + b$, which is the token's net contribution and has the sign of $a$.
    - Otherwise $b$ is at least as large as $a$ in magnitude and cancels all of it, so nothing of
      $a$ remains.
    """
    # fmt: off
    return (
        pl.when(are_same_sign(component1, component2)).then(
            component1
        ).otherwise(
            pl.when(component1.abs() > component2.abs()).then(
                component1 + component2
            ).otherwise(
                0
            )
        )
    )
    # fmt: on


def plot_component_counteracted(component1: pl.Expr, component2: pl.Expr) -> pl.Expr:
    r"""
    Computes how much of `component1` is cancelled by `component2`, the other component of the
    same contribution.

    It is the complement of `plot_component`, which computes what remains of `component1` rather
    than what is cancelled. The two always sum to `component1`.

    With $a$ as `component1` and $b$ as `component2`:

    $$
    \begin{cases}
        0 & a \text{ and } b \text{ share a sign} \\
        -b & |a| > |b| \\
        a & \text{otherwise}
    \end{cases}
    $$

    - If $a$ and $b$ share a sign, both push the contribution in the same direction, so none of
      $a$ is cancelled.
    - If they have opposite signs and $a$ is larger in magnitude, $b$ cancels as much of $a$ as
      its own magnitude, which is $-b$ and has the sign of $a$.
    - Otherwise $b$ is at least as large as $a$ in magnitude, so all of $a$ is cancelled.
    """
    # fmt: off
    return (
        pl.when(are_same_sign(component1, component2)).then(
            0
        ).otherwise(
            pl.when(component1.abs() > component2.abs()).then(
                -1 * component2
            ).otherwise(
                component1
            )
        )
    )
    # fmt: on


def conditional_plot_component(
    component1: pl.Expr,
    component2: pl.Expr,
    condition: pl.Expr,
    counteracted: bool,
) -> pl.Expr:
    """
    Computes one bar of a word shift graph: what remains of `component1` after `component2`
    cancels against it, for the tokens that meet `condition`, and 0 for every other token.

    If `counteracted`, computes how much of `component1` is cancelled by `component2` instead.
    """
    if counteracted:
        plot_component_func = plot_component_counteracted
    else:
        plot_component_func = plot_component

    return pl.when(condition).then(plot_component_func(component1, component2)).otherwise(0)


def more_of_positive(
    frequency_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    counteracted: bool,
) -> pl.Expr:
    r"""
    Computes the bar for a relatively positive token that is used more, $(+\uparrow)$. It applies
    to tokens whose `frequency_diff` and `reference_score_diff` are both non-negative.
    """
    return conditional_plot_component(
        component1=frequency_diff * reference_score_diff,
        component2=weighted_score_diff,
        condition=(frequency_diff >= 0) & (reference_score_diff >= 0),
        counteracted=counteracted,
    )


def less_of_negative(
    frequency_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    counteracted: bool,
) -> pl.Expr:
    r"""
    Computes the bar for a relatively negative token that is used less, $(-\downarrow)$. It applies
    to tokens whose `frequency_diff` and `reference_score_diff` are both negative.
    """
    return conditional_plot_component(
        component1=frequency_diff * reference_score_diff,
        component2=weighted_score_diff,
        condition=(frequency_diff < 0) & (reference_score_diff < 0),
        counteracted=counteracted,
    )


def more_of_negative(
    frequency_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    counteracted: bool,
) -> pl.Expr:
    r"""
    Computes the bar for a relatively negative token that is used more, $(-\uparrow)$. It applies
    to tokens whose `frequency_diff` is non-negative and whose `reference_score_diff` is negative.
    """
    return conditional_plot_component(
        component1=frequency_diff * reference_score_diff,
        component2=weighted_score_diff,
        condition=(frequency_diff >= 0) & (reference_score_diff < 0),
        counteracted=counteracted,
    )


def less_of_positive(
    frequency_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    counteracted: bool,
) -> pl.Expr:
    r"""
    Computes the bar for a relatively positive token that is used less, $(+\downarrow)$. It applies
    to tokens whose `frequency_diff` is negative and whose `reference_score_diff` is non-negative.
    """
    return conditional_plot_component(
        component1=frequency_diff * reference_score_diff,
        component2=weighted_score_diff,
        condition=(frequency_diff < 0) & (reference_score_diff >= 0),
        counteracted=counteracted,
    )


def higher_score(
    frequency_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    counteracted: bool,
) -> pl.Expr:
    r"""
    Computes the bar for a token whose score is higher in the second corpus, $(\bigtriangleup)$. It
    applies to tokens whose `weighted_score_diff` is non-negative.
    """
    return conditional_plot_component(
        component1=weighted_score_diff,
        component2=frequency_diff * reference_score_diff,
        condition=(weighted_score_diff >= 0),
        counteracted=counteracted,
    )


def lower_score(
    frequency_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    counteracted: bool,
) -> pl.Expr:
    r"""
    Computes the bar for a token whose score is lower in the second corpus, $(\bigtriangledown)$.
    It applies to tokens whose `weighted_score_diff` is negative.
    """
    return conditional_plot_component(
        component1=weighted_score_diff,
        component2=frequency_diff * reference_score_diff,
        condition=(weighted_score_diff < 0),
        counteracted=counteracted,
    )
