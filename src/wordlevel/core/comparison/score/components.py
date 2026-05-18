from typing import Literal, Optional

import polars as pl
from polars import col

from wordlevel.core.utils import are_same_sign, normalize_freqs


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
    """ """
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
    """ """
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
    """ """
    probs1 = normalize_freqs(freqs1, normalize=normalize)
    probs2 = normalize_freqs(freqs2, normalize=normalize)

    return (0.5 * probs1 + 0.5 * probs2) * (scores2 - scores1)


def reference_score_diff(
    scores1: pl.Expr,
    scores2: pl.Expr,
    reference_score: float | Literal["mean"] | pl.Expr,
    freqs1: Optional[pl.Expr] = None,
    normalize: bool = True,
) -> pl.Expr:
    """ """
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
    """ """
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
    """ """
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
    """ """
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
    """ """
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
    """ """
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
    """ """
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
    """ """
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
    """ """
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
    """ """
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
    """ """
    return conditional_plot_component(
        component1=weighted_score_diff,
        component2=frequency_diff * reference_score_diff,
        condition=(weighted_score_diff < 0),
        counteracted=counteracted,
    )
