from typing import Literal, Optional

import polars as pl

from wordlevel.types import PolarsFrame
from wordlevel.expressions.utils import normalize_freqs


REFERENCE_ATTRIBUTION = "reference"
COMPARISON_ATTRIBUTION = "comparison"
ATTRIBUTIONS = pl.Enum([REFERENCE_ATTRIBUTION, COMPARISON_ATTRIBUTION])

ATTRIBUTION_TYPE = Literal["frequency", "rank", "score_sign"]


def frequency_attribution(
    reference: pl.Expr, comparison: pl.Expr, normalize: bool = True
) -> pl.Expr:
    """ """
    if normalize:
        reference = normalize_freqs(reference, normalize=normalize)
        comparison = normalize_freqs(comparison, normalize=normalize)

    return (
        pl.when(comparison >= reference)
        .then(pl.lit(COMPARISON_ATTRIBUTION))
        .otherwise(pl.lit(REFERENCE_ATTRIBUTION))
        .cast(ATTRIBUTIONS)
    )


def rank_attribution(reference: pl.Expr, comparison: pl.Expr) -> pl.Expr:
    """ """
    reference_rank = reference.rank(descending=True)
    comparison_rank = comparison.rank(descending=True)

    return (
        pl.when(comparison_rank <= reference_rank)
        .then(pl.lit(COMPARISON_ATTRIBUTION))
        .otherwise(pl.lit(REFERENCE_ATTRIBUTION))
        .cast(ATTRIBUTIONS)
    )


def score_sign_attribution(score: pl.Expr, comparison_is_positive: bool = True) -> pl.Expr:
    """ """
    if comparison_is_positive:
        greater_than_attribution = COMPARISON_ATTRIBUTION
        less_than_attribution = REFERENCE_ATTRIBUTION
    else:
        greater_than_attribution = REFERENCE_ATTRIBUTION
        less_than_attribution = COMPARISON_ATTRIBUTION

    return (
        pl.when(score >= 0)
        .then(pl.lit(greater_than_attribution))
        .otherwise(pl.lit(less_than_attribution))
        .cast(ATTRIBUTIONS)
    )


def attribute_scores(
    df: PolarsFrame,
    by: ATTRIBUTION_TYPE,
    reference: Optional[pl.Expr] = None,
    comparison: Optional[pl.Expr] = None,
    score: Optional[pl.Expr] = None,
    normalize: bool = True,
    attribution_alias: str = "attribution",
) -> PolarsFrame:
    """ """
    if by in {"frequency", "rank"} and (reference is None or comparison is None):
        raise ValueError(f"Need reference and comparison to attribute by {by}")
    elif by == "score_sign" and score is None:
        raise ValueError("Need score to attribute by score_sign")

    if by == "frequency":
        expr = frequency_attribution(
            reference=reference, comparison=comparison, normalize=normalize
        )
    elif by == "rank":
        expr = rank_attribution(reference=reference, comparison=comparison)
    elif by == "score_sign":
        expr = score_sign_attribution(score=score)
    else:
        raise ValueError("Invalid specification of how to attribute scores")

    return df.with_columns(expr.alias(attribution_alias))
