from typing import Literal

import polars as pl

from wordlevel.constants import (
    COMPARISON_ATTRIBUTION,
    OFFSET_ATTRIBUTION,
    REFERENCE_ATTRIBUTION,
    REINFORCE_ATTRIBUTION,
)
from wordlevel.utils import are_same_sign, normalize_freqs


ATTRIBUTION_TYPE = Literal["frequency", "rank", "score_sign"]


ATTRIBUTION_LABELS: dict[ATTRIBUTION_TYPE, tuple[str, str]] = {
    "frequency": (REFERENCE_ATTRIBUTION, COMPARISON_ATTRIBUTION),
    "rank": (REFERENCE_ATTRIBUTION, COMPARISON_ATTRIBUTION),
    "score_sign": (REINFORCE_ATTRIBUTION, OFFSET_ATTRIBUTION),
}
"""
Single source of truth for the pair of attribution labels each attribution type produces
"""


def frequency_attribution(
    reference: pl.Expr,
    comparison: pl.Expr,
    normalize: bool = True,
    reference_attribution: str = REFERENCE_ATTRIBUTION,
    comparison_attribution: str = COMPARISON_ATTRIBUTION,
) -> pl.Expr:
    """
    Attributes each token to the corpus that uses it more, labeled `reference_attribution` or
    `comparison_attribution`. A token used equally in both corpora is attributed to the comparison
    corpus. If `normalize` is `False`, the frequencies are compared as given rather than normalized
    to sum to 1.
    """
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
    """
    Attributes each token to the corpus that ranks it higher, labeled `reference_attribution` or
    `comparison_attribution`. Rank 1 is the most frequent token of a corpus, and tokens of equal
    frequency share the mean of the ranks they span. A token ranked equally in both corpora is
    attributed to the comparison corpus.
    """
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
    opposite sign offsets the difference, shrinking its magnitude. A contribution of 0, and every
    contribution when the total is 0, is attributed as reinforcing.
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
    reference: pl.Expr | None = None,
    comparison: pl.Expr | None = None,
    score: pl.Expr | None = None,
    normalize: bool = True,
    labels: tuple[str, str] | None = None,
    attribution_alias: str = "attribution",
) -> pl.DataFrame | pl.LazyFrame:
    """
    Attributes each token's contribution using the strategy named by `by`.

    Args:
        df:
            The dataframe of tokens to attribute
        by:
            How each contribution is attributed. `"frequency"` and `"rank"` attribute it to the
            corpus that uses the token more or ranks it higher, and `"score_sign"` to whether it
            reinforces or offsets the overall difference
        reference:
            Frequency of each token in the reference corpus. Required when `by` is `"frequency"`
            or `"rank"`
        comparison:
            Frequency of each token in the comparison corpus. Required when `by` is
            `"frequency"` or `"rank"`
        score:
            Each token's contribution. Required when `by` is `"score_sign"`
        normalize:
            Whether to normalize the frequencies to sum to 1 before comparing them. Only applies
            when `by` is `"frequency"`
        labels:
            The pair of labels for the two attributions, overriding the defaults for `by` in
            `ATTRIBUTION_LABELS`
        attribution_alias:
            Name to give the new column holding the attribution

    Returns:
        `df` with the attribution column added

    Raises:
        ValueError:
            If `by` is `"frequency"` or `"rank"` and `reference` or `comparison` is not given.
        ValueError: If `by` is `"score_sign"` and `score` is not given.
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
