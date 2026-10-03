from collections.abc import Mapping
from typing import Literal, Protocol

import polars as pl
from polars import col

from wordlevel.constants import IS_BORROWED_COL
from wordlevel.core.comparison.score.attribution import ATTRIBUTION_TYPE, attribute_scores
from wordlevel.core.comparison.score.components import (
    get_shift_components,
    get_shift_plot_components,
)
from wordlevel.core.comparison.score.lexicon import (
    DEFAULT_LEXICON_SCORE_COL,
    ScoredInOneLexicon,
    get_imputation_expression,
    join_lexicon,
)
from wordlevel.lexicons.core.load import Lexicon


class ScoreFunc(Protocol):
    def __call__(
        self,
        freqs1: pl.Expr,
        freqs2: pl.Expr,
    ) -> pl.Expr: ...


class DualScoreFunc(Protocol):
    def __call__(
        self, freqs1: pl.Expr, freqs2: pl.Expr, alias1: str, alias2: str
    ) -> tuple[pl.Expr, pl.Expr]: ...


class DualCleanScoreFunc(Protocol):
    def __call__(
        self,
        freqs1: pl.Expr,
        freqs2: pl.Expr,
        scores1: pl.Expr,
        scores2: pl.Expr,
        alias1: str,
        alias2: str,
    ) -> tuple[pl.Expr, pl.Expr]: ...


def basic_score_plan(
    df: pl.DataFrame | pl.LazyFrame,
    freqs_ref: pl.Expr,
    freqs_cmp: pl.Expr,
    token: pl.Expr,
    score_func: ScoreFunc,
    attribute_by: ATTRIBUTION_TYPE,
    score_alias: str = "comparison_score",
    rank_alias: str = "rank",
    attribution_alias: str = "attribution",
) -> pl.DataFrame | pl.LazyFrame:
    """ """
    attribution_cols = {}
    if attribute_by == "score_sign":
        attribution_cols["score"] = col(score_alias)
    else:
        attribution_cols["reference"] = freqs_ref
        attribution_cols["comparison"] = freqs_cmp

    df_plan = (
        df.select(
            token,
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
            attribution_alias=attribution_alias,
        )
    )

    return df_plan


def weighted_average_score_plan(
    df: pl.DataFrame | pl.LazyFrame,
    score_freqs_ref: pl.Expr,
    score_freqs_cmp: pl.Expr,
    component_freqs_ref: pl.Expr,
    component_freqs_cmp: pl.Expr,
    token: pl.Expr,
    score_func: DualScoreFunc,
    clean_score_func: DualCleanScoreFunc,
    attribute_by: ATTRIBUTION_TYPE,
    reference_score: float | Literal["mean"] | pl.Expr = 0,
    passthrough_cols: list[str] | None = None,
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
) -> pl.DataFrame | pl.LazyFrame:
    """ """
    attribution_cols = {}
    if attribute_by == "score_sign":
        attribution_cols["score"] = col(score_alias)
    else:
        attribution_cols["reference"] = score_freqs_ref
        attribution_cols["comparison"] = score_freqs_cmp

    # Select only the score columns for the reference and comparison corpora (and not all others).
    # Dedupe by output name so component freqs that already appear (e.g. when one side is shared
    # with score freqs) aren't selected twice — this triggers a polars DuplicateError.
    score_cols_to_select: list[pl.Expr] = []
    seen_names: set[str] = set()
    for expr in (score_freqs_ref, score_freqs_cmp, component_freqs_ref, component_freqs_cmp):
        name = expr.meta.output_name()
        if name not in seen_names:
            score_cols_to_select.append(expr)
            seen_names.add(name)

    if passthrough_cols is not None:
        score_cols_to_select.extend([col(c) for c in passthrough_cols])

    df_plan = (
        df.select(
            token,
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
            attribution_alias=attribution_alias,
        )
        .drop("clean_score_ref", "clean_score_cmp")
        .rename({"raw_score_ref": "score_ref", "raw_score_cmp": "score_cmp"})
    )

    return df_plan


def lexicon_join_plan(
    df: pl.DataFrame | pl.LazyFrame,
    df_token_col: str,
    lexicon: Lexicon | Mapping[str, float] | pl.DataFrame | pl.LazyFrame | None = None,
    lexicon_ref: Lexicon | Mapping[str, float] | pl.DataFrame | pl.LazyFrame | None = None,
    lexicon_cmp: Lexicon | Mapping[str, float] | pl.DataFrame | pl.LazyFrame | None = None,
    lexicon_token_col: str | None = None,
    lexicon_score_col: str | None = None,
    score_alias_ref: str | None = None,
    score_alias_cmp: str | None = None,
    scored_in_one_lexicon: float | ScoredInOneLexicon | pl.Expr = "drop",
    drop_null_scores: bool = True,
) -> pl.DataFrame | pl.LazyFrame:
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

    impute_strategy: float | Literal["mean"] | pl.Expr | None = None
    if not isinstance(scored_in_one_lexicon, str) or scored_in_one_lexicon == "mean":
        impute_strategy = scored_in_one_lexicon

    df = join_lexicon(
        df,
        lexicon_ref,  # type: ignore[arg-type]
        df_token_col=df_token_col,
        lexicon_token_col=lexicon_token_col,
        lexicon_score_col=lexicon_score_col,
    ).rename({score_col_to_rename: score_alias_ref})
    df = join_lexicon(
        df,
        lexicon_cmp,  # type: ignore[arg-type]
        df_token_col=df_token_col,
        lexicon_token_col=lexicon_token_col,
        lexicon_score_col=lexicon_score_col,
    ).rename({score_col_to_rename: score_alias_cmp})

    # Only tokens scored by exactly one lexicon are imputed; tokens in neither stay null
    if impute_strategy is not None:
        df = df.with_columns(
            pl.when(col(score_alias_ref).is_null() & col(score_alias_cmp).is_not_null())
            .then(
                get_imputation_expression(
                    lexicon_ref,  # type: ignore[arg-type]
                    impute=impute_strategy,
                    lexicon_score_col=lexicon_score_col,
                )
            )
            .otherwise(col(score_alias_ref))
            .alias(score_alias_ref),
            pl.when(col(score_alias_cmp).is_null() & col(score_alias_ref).is_not_null())
            .then(
                get_imputation_expression(
                    lexicon_cmp,  # type: ignore[arg-type]
                    impute=impute_strategy,
                    lexicon_score_col=lexicon_score_col,
                )
            )
            .otherwise(col(score_alias_cmp))
            .alias(score_alias_cmp),
        )

    if isinstance(scored_in_one_lexicon, str) and scored_in_one_lexicon == "borrow":
        is_borrowed = (col(score_alias_ref).is_null() & col(score_alias_cmp).is_not_null()) | (
            col(score_alias_cmp).is_null() & col(score_alias_ref).is_not_null()
        )

        df = df.with_columns(
            pl.when(col(score_alias_ref).is_null())
            .then(col(score_alias_cmp))
            .otherwise(col(score_alias_ref))
            .alias(score_alias_ref),
            pl.when(col(score_alias_cmp).is_null())
            .then(col(score_alias_ref))
            .otherwise(col(score_alias_cmp))
            .alias(score_alias_cmp),
            is_borrowed.alias(IS_BORROWED_COL),
        )

    if drop_null_scores:
        df = df.filter(col(score_alias_ref).is_not_null() & col(score_alias_cmp).is_not_null())

    return df
