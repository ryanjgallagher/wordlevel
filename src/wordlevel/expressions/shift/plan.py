from typing import Literal

import polars as pl

from polars import col

from wordlevel.types import PolarsFrame
from wordlevel.expressions.types import ScoreFunc, DualScoreFunc, DualCleanScoreFunc
from wordlevel.expressions.attribution import attribute_scores, ATTRIBUTION_TYPE
from wordlevel.expressions.shift.components import get_shift_components, get_shift_plot_components


def basic_score_plan(
    df: PolarsFrame,
    freqs_ref: pl.Expr,
    freqs_cmp: pl.Expr,
    score_func: ScoreFunc,
    attribute_by: ATTRIBUTION_TYPE,
    score_alias: str = "comparison_score",
    rank_alias: str = "rank",
    attribution_alias: str = "attribution",
    reference_attribution: str = "reference",
    comparison_attribution: str = "comparison",
) -> PolarsFrame:
    """
    TODO: no magic strings
    """
    attribution_cols = dict()
    if attribute_by == "score_sign":
        attribution_cols["score"] = col(score_alias)
    else:
        attribution_cols["reference"] = freqs_ref
        attribution_cols["comparison"] = freqs_cmp

    df_plan = (
        df.with_columns(score_func(freqs1=freqs_ref, freqs2=freqs_cmp).alias(score_alias))
        .with_columns(col(score_alias).abs().rank(descending=True).alias(rank_alias))
        .pipe(
            attribute_scores,
            **attribution_cols,
            by=attribute_by,
            normalize=False,
            reference_attribution=reference_attribution,
            comparison_attribution=comparison_attribution,
            attribution_alias=attribution_alias,
        )
    )

    return df_plan


def weighted_average_score_plan(
    df: PolarsFrame,
    score_freqs_ref: pl.Expr,
    score_freqs_cmp: pl.Expr,
    component_freqs_ref: pl.Expr,
    component_freqs_cmp: pl.Expr,
    score_func: DualScoreFunc,
    clean_score_func: DualCleanScoreFunc,
    attribute_by: ATTRIBUTION_TYPE,
    reference_score: float | Literal["mean"] | pl.Expr = 0,
    score_alias: str = "comparison_score",
    rank_alias: str = "rank",
    attribution_alias: str = "attribution",
    reference_attribution: str = "reference",
    comparison_attribution: str = "comparison",
) -> PolarsFrame:
    """
    TODO: no magic strings
    """
    attribution_cols = dict()
    if attribute_by == "score_sign":
        attribution_cols["score"] = col(score_alias)
    else:
        attribution_cols["reference"] = score_freqs_ref
        attribution_cols["comparison"] = score_freqs_cmp

    df_plan = (
        df.with_columns(
            *score_func(
                freqs1=score_freqs_ref,
                freqs2=score_freqs_cmp,
                alias1="raw_score_ref",
                alias2="raw_score_cmp",
            ),
        )
        .with_columns(
            *clean_score_func(
                freqs1=score_freqs_ref,
                freqs2=score_freqs_cmp,
                scores1=col("raw_score_ref"),
                scores2=col("raw_score_cmp"),
                alias1="clean_score_ref",
                alias2="clean_score_cmp",
            )
        )
        .pipe(
            get_shift_components,
            freqs1=component_freqs_ref,
            freqs2=component_freqs_cmp,
            scores1=col("clean_score_ref"),
            scores2=col("clean_score_cmp"),
            reference_score=reference_score,
            normalize=False,
            include_total_difference=True,
            include_rank=True,
            frequency_diff_alias="freq_diff",
            weighted_score_diff_alias="weighted_score_diff",
            reference_score_diff_alias="ref_score_diff",
            total_difference_alias=score_alias,
            rank_alias=rank_alias,
        )
        .pipe(
            get_shift_plot_components,
            frequency_diff=col("freq_diff"),
            weighted_score_diff=col("weighted_score_diff"),
            reference_score_diff=col("ref_score_diff"),
        )
        .pipe(
            attribute_scores,
            **attribution_cols,
            by=attribute_by,
            normalize=False,
            reference_attribution=reference_attribution,
            comparison_attribution=comparison_attribution,
            attribution_alias=attribution_alias,
        )
        .drop("clean_score_ref", "clean_score_cmp")
        .rename({"raw_score_ref": "score_ref", "raw_score_cmp": "score_cmp"})
    )

    return df_plan
