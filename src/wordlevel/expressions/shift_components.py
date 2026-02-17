from typing import Literal, Optional

import polars as pl
from polars import col

from wordlevel.types import PolarsFrame
from wordlevel.expressions.utils import normalize_freqs


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


def get_shift_components(
    df: PolarsFrame,
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    scores1: pl.Expr,
    scores2: pl.Expr,
    reference_score: float,
    normalize: bool = True,
    include_total_difference: bool = True,
    include_rank: bool = False,
    frequency_diff_alias: str = "freq_diff",
    weighted_score_diff_alias: str = "weighted_score_diff",
    reference_score_diff_alias: str = "ref_score_diff",
    total_difference_alias: str = "total_difference",
    rank_alias: str = "rank",
) -> PolarsFrame:
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
            .alias(total_difference_alias)
        )

        if include_rank:
            df_comps = df_comps.with_columns(
                col(total_difference_alias).abs().rank(descending=True).alias(rank_alias)
            )
    # fmt: on

    return df_comps
