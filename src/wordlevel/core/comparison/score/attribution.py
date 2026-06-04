from typing import Literal, Optional

import polars as pl

from wordlevel.core.utils import are_same_sign, normalize_freqs


ATTRIBUTION_TYPE = Literal["frequency", "rank", "score_sign"]


def frequency_attribution(
    reference: pl.Expr,
    comparison: pl.Expr,
    normalize: bool = True,
    reference_attribution: str = "reference",
    comparison_attribution: str = "comparison",
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
    reference_attribution: str = "reference",
    comparison_attribution: str = "comparison",
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
    reference_attribution: str = "reference",
    comparison_attribution: str = "comparison",
) -> pl.Expr:
    """ """
    return (
        pl.when(are_same_sign(score, score.sum()))
        .then(pl.lit(comparison_attribution))
        .otherwise(pl.lit(reference_attribution))
        .cast(pl.Enum([reference_attribution, comparison_attribution]))
    )


def attribute_scores(
    df: pl.DataFrame | pl.LazyFrame,
    by: ATTRIBUTION_TYPE,
    reference: Optional[pl.Expr] = None,
    comparison: Optional[pl.Expr] = None,
    score: Optional[pl.Expr] = None,
    normalize: bool = True,
    reference_attribution: str = "reference",
    comparison_attribution: str = "comparison",
    attribution_alias: str = "attribution",
) -> pl.DataFrame | pl.LazyFrame:
    """ """
    if by == "frequency":
        if reference is None or comparison is None:
            raise ValueError(f"Need reference and comparison to attribute by {by}")
        expr = frequency_attribution(
            reference=reference,
            comparison=comparison,
            normalize=normalize,
            reference_attribution=reference_attribution,
            comparison_attribution=comparison_attribution,
        )
    elif by == "rank":
        if reference is None or comparison is None:
            raise ValueError(f"Need reference and comparison to attribute by {by}")
        expr = rank_attribution(
            reference=reference,
            comparison=comparison,
            reference_attribution=reference_attribution,
            comparison_attribution=comparison_attribution,
        )
    else:
        if score is None:
            raise ValueError("Need score to attribute by score_sign")
        expr = score_sign_attribution(
            score=score,
            reference_attribution=reference_attribution,
            comparison_attribution=comparison_attribution,
        )

    return df.with_columns(expr.alias(attribution_alias))
