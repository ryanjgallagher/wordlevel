from typing import Literal, Mapping

import polars as pl

from polars import col

from wordlevel.types import PolarsFrame
from wordlevel.expressions.types import ScoreFunc, DualScoreFunc, DualCleanScoreFunc
from wordlevel.expressions.attribution import attribute_scores, ATTRIBUTION_TYPE
from wordlevel.expressions.shift.components import get_shift_components, get_shift_plot_components
from wordlevel.expressions.lexicon import join_lexicon, DEFAULT_LEXICON_SCORE_COL
from wordlevel.lexicons.lexicon import Lexicon


def basic_score_plan(
    df: PolarsFrame,
    freqs_ref: pl.Expr,
    freqs_cmp: pl.Expr,
    label: pl.Expr,
    score_func: ScoreFunc,
    attribute_by: ATTRIBUTION_TYPE,
    score_alias: str = "comparison_score",
    rank_alias: str = "rank",
    attribution_alias: str = "attribution",
    reference_attribution: str = "reference",
    comparison_attribution: str = "comparison",
) -> PolarsFrame:
    """ """
    attribution_cols = dict()
    if attribute_by == "score_sign":
        attribution_cols["score"] = col(score_alias)
    else:
        attribution_cols["reference"] = freqs_ref
        attribution_cols["comparison"] = freqs_cmp

    df_plan = (
        df.select(
            label,
            freqs_ref,
            freqs_cmp,
            score_func(freqs1=freqs_ref, freqs2=freqs_cmp).alias(score_alias),
        )
        .with_columns(
            col(score_alias).abs().rank(descending=True, method="ordinal").alias(rank_alias)
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
    )

    return df_plan


def weighted_average_score_plan(
    df: PolarsFrame,
    score_freqs_ref: pl.Expr,
    score_freqs_cmp: pl.Expr,
    component_freqs_ref: pl.Expr,
    component_freqs_cmp: pl.Expr,
    label: pl.Expr,
    score_func: DualScoreFunc,
    clean_score_func: DualCleanScoreFunc,
    attribute_by: ATTRIBUTION_TYPE,
    reference_score: float | Literal["mean"] | pl.Expr = 0,
    score_alias: str = "comparison_score",
    rank_alias: str = "rank",
    attribution_alias: str = "attribution",
    more_of_positive_alias: str = "more_of_positive",
    less_of_negative_alias: str = "less_of_negative",
    more_of_negative_alias: str = "more_of_negative",
    less_of_positive_alias: str = "less_of_positive",
    higher_score_alias: str = "higher_score",
    lower_score_alias: str = "lower_score",
    counteracted_suffix: str = "_counteracted",
    reference_attribution: str = "reference",
    comparison_attribution: str = "comparison",
) -> PolarsFrame:
    """ """
    attribution_cols = dict()
    if attribute_by == "score_sign":
        attribution_cols["score"] = col(score_alias)
    else:
        attribution_cols["reference"] = score_freqs_ref
        attribution_cols["comparison"] = score_freqs_cmp

    # Select only the score columns for the reference and comparison corpora (and not all others)
    # Avoid duplicate column selection error if the component freqs are the same as the score freqs
    score_cols_to_select = [score_freqs_ref, score_freqs_cmp]
    if score_freqs_ref.meta.output_name() != component_freqs_ref.meta.output_name():
        score_cols_to_select += [component_freqs_ref, component_freqs_cmp]

    df_plan = (
        df.select(
            label,
            *score_cols_to_select,
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
            more_of_positive_alias=more_of_positive_alias,
            less_of_negative_alias=less_of_negative_alias,
            more_of_negative_alias=more_of_negative_alias,
            less_of_positive_alias=less_of_positive_alias,
            higher_score_alias=higher_score_alias,
            lower_score_alias=lower_score_alias,
            counteracted_suffix=counteracted_suffix,
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


def lexicon_join_plan(
    df: PolarsFrame,
    df_label_col: str,
    lexicon: Lexicon | Mapping[str, float] | PolarsFrame | None = None,
    lexicon_ref: Lexicon | Mapping[str, float] | PolarsFrame | None = None,
    lexicon_cmp: Lexicon | Mapping[str, float] | PolarsFrame | None = None,
    lexicon_label_col: str | None = None,
    lexicon_score_col: str | None = None,
    score_alias_ref: str | None = None,
    score_alias_cmp: str | None = None,
    impute: float | Literal["mean"] | pl.Expr = 0,
) -> PolarsFrame:
    """ """
    if lexicon is None and lexicon_ref is None and lexicon_cmp is None:
        raise ValueError("Must specify at least one of the lexicon parameters")
    if lexicon is not None and (lexicon_ref is not None or lexicon_cmp is not None):
        raise ValueError(
            "`lexicon` is ambiguous when the reference or comparison lexicons are also specified"
        )

    if lexicon is not None:
        lexicon_ref = lexicon
        lexicon_cmp = lexicon
    else:
        if lexicon_ref is None:
            lexicon_ref = lexicon_cmp
        if lexicon_cmp is None:
            lexicon_cmp = lexicon_ref

    if lexicon_score_col is not None:
        score_col_to_rename = lexicon_score_col
    elif isinstance(lexicon_ref, Lexicon):
        score_col_to_rename = lexicon_ref.score_col
    else:
        score_col_to_rename = DEFAULT_LEXICON_SCORE_COL

    if score_alias_ref is None:
        score_alias_ref = f"{score_col_to_rename}_ref"
    if score_alias_cmp is None:
        score_alias_cmp = f"{score_col_to_rename}_cmp"

    df = join_lexicon(
        df,
        lexicon_ref,
        df_label_col=df_label_col,
        lexicon_label_col=lexicon_label_col,
        lexicon_score_col=lexicon_score_col,
        impute=impute,
    ).rename({score_col_to_rename: score_alias_ref})
    df = join_lexicon(
        df,
        lexicon_cmp,
        df_label_col=df_label_col,
        lexicon_label_col=lexicon_label_col,
        lexicon_score_col=lexicon_score_col,
        impute=impute,
    ).rename({score_col_to_rename: score_alias_cmp})

    return df
