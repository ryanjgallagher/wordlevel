from typing import Literal, Optional

import polars as pl

from wordlevel.constants import (
    COMPARISON_ATTRIBUTION,
    OFFSET_ATTRIBUTION,
    REFERENCE_ATTRIBUTION,
    REINFORCE_ATTRIBUTION,
)
from wordlevel.core.utils import are_same_sign, normalize_freqs


ATTRIBUTION_TYPE = Literal["frequency", "rank", "score_sign"]


ATTRIBUTION_LABELS: dict[ATTRIBUTION_TYPE, tuple[str, str]] = {
    "frequency": (REFERENCE_ATTRIBUTION, COMPARISON_ATTRIBUTION),
    "rank": (REFERENCE_ATTRIBUTION, COMPARISON_ATTRIBUTION),
    "score_sign": (REINFORCE_ATTRIBUTION, OFFSET_ATTRIBUTION),
}
"""
Single source of truth for the pair of attribution labels each attribution type produces. The tuple
order matches the field order of the corresponding bar-config class (`BarConfig(reference,
comparison)` for corpus attribution, `ScoreSignBarConfig(reinforce, offset)` for score-sign
attribution).
"""


def frequency_attribution(
    reference: pl.Expr,
    comparison: pl.Expr,
    normalize: bool = True,
    reference_attribution: str = REFERENCE_ATTRIBUTION,
    comparison_attribution: str = COMPARISON_ATTRIBUTION,
) -> pl.Expr:
    """ """
    if normalize:
        reference = normalize_freqs(reference, normalize=normalize)
        comparison = normalize_freqs(comparison, normalize=normalize)

    return (
        pl.when(comparison >= reference)
        .then(pl.lit(comparison_attribution))
        .otherwise(pl.lit(reference_attribution))
        .cast(pl.Enum([reference_attribution, comparison_attribution]))
    )


def rank_attribution(
    reference: pl.Expr,
    comparison: pl.Expr,
    reference_attribution: str = REFERENCE_ATTRIBUTION,
    comparison_attribution: str = COMPARISON_ATTRIBUTION,
) -> pl.Expr:
    """ """
    reference_rank = reference.rank(descending=True)
    comparison_rank = comparison.rank(descending=True)

    return (
        pl.when(comparison_rank <= reference_rank)
        .then(pl.lit(comparison_attribution))
        .otherwise(pl.lit(reference_attribution))
        .cast(pl.Enum([reference_attribution, comparison_attribution]))
    )


def score_sign_attribution(
    score: pl.Expr,
    reinforce_attribution: str = REINFORCE_ATTRIBUTION,
    offset_attribution: str = OFFSET_ATTRIBUTION,
) -> pl.Expr:
    """
    Attributes each token by whether its contribution reinforces or offsets the overall difference.

    A token whose contribution has the same sign as the total (the sum of all contributions)
    reinforces the difference, making its magnitude larger. A token whose contribution has the
    opposite sign offsets the difference, shrinking its magnitude.
    """
    return (
        pl.when(are_same_sign(score, score.sum()))
        .then(pl.lit(reinforce_attribution))
        .otherwise(pl.lit(offset_attribution))
        .cast(pl.Enum([reinforce_attribution, offset_attribution]))
    )


def attribute_scores(
    df: pl.DataFrame | pl.LazyFrame,
    by: ATTRIBUTION_TYPE,
    reference: Optional[pl.Expr] = None,
    comparison: Optional[pl.Expr] = None,
    score: Optional[pl.Expr] = None,
    normalize: bool = True,
    labels: Optional[tuple[str, str]] = None,
    attribution_alias: str = "attribution",
) -> pl.DataFrame | pl.LazyFrame:
    """
    Attributes each token's contribution using the strategy named by `by`.

    The pair of labels used to name the two attribution buckets is taken from `ATTRIBUTION_LABELS`
    for the given `by`, unless `labels` is provided to override them.
    """
    label_first, label_second = labels if labels is not None else ATTRIBUTION_LABELS[by]

    if by == "frequency":
        if reference is None or comparison is None:
            raise ValueError(f"Need reference and comparison to attribute by {by}")
        expr = frequency_attribution(
            reference=reference,
            comparison=comparison,
            normalize=normalize,
            reference_attribution=label_first,
            comparison_attribution=label_second,
        )
    elif by == "rank":
        if reference is None or comparison is None:
            raise ValueError(f"Need reference and comparison to attribute by {by}")
        expr = rank_attribution(
            reference=reference,
            comparison=comparison,
            reference_attribution=label_first,
            comparison_attribution=label_second,
        )
    else:
        if score is None:
            raise ValueError("Need score to attribute by score_sign")
        expr = score_sign_attribution(
            score=score,
            reinforce_attribution=label_first,
            offset_attribution=label_second,
        )

    return df.with_columns(expr.alias(attribution_alias))
