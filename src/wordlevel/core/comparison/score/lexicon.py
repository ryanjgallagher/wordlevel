from typing import Literal, Mapping

import polars as pl
from polars import col

from wordlevel.api.lexicons.lexicon import Lexicon


DEFAULT_LEXICON_SCORE_COL = "lexicon_score"


def join_lexicon(
    df: pl.DataFrame | pl.LazyFrame,
    lexicon: Lexicon | Mapping[str, float] | pl.DataFrame | pl.LazyFrame,
    df_token_col: str,
    lexicon_token_col: str | None = None,
    lexicon_score_col: str | None = None,
    impute: float | Literal["mean"] | pl.Expr | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """ """
    if isinstance(lexicon, Lexicon):
        df_scores = _join_lexicon_from_dataframe(
            df=df.lazy(),
            df_lexicon=lexicon.lf,
            df_token_col=df_token_col,
            lexicon_token_col=lexicon.token_col,
            lexicon_score_col=lexicon.score_col,
            impute=impute,
        )
    elif isinstance(lexicon, (pl.DataFrame, pl.LazyFrame)):
        if lexicon_token_col is None:
            raise ValueError("Must specify `lexicon_token_col` if the lexicon is a dataframe")
        elif lexicon_score_col is None:
            raise ValueError("Must specify `lexicon_score_col` if the lexicon is a dataframe")

        df_scores = _join_lexicon_from_dataframe(
            df=df.lazy(),
            df_lexicon=lexicon.lazy(),
            df_token_col=df_token_col,
            lexicon_token_col=lexicon_token_col,
            lexicon_score_col=lexicon_score_col,
            impute=impute,
        )
    elif isinstance(lexicon, Mapping):
        if lexicon_score_col is None:
            lexicon_score_col = DEFAULT_LEXICON_SCORE_COL

        df_scores = _join_lexicon_from_mapping(
            df=df.lazy(),
            lexicon=lexicon,
            df_token_col=df_token_col,
            lexicon_score_alias=lexicon_score_col,
            impute=impute,
        )
    else:
        raise ValueError(
            f" The lexicon must be a Mapping or Polars DataFrame/LazyFrame, got: {type(lexicon)}"
        )

    if isinstance(df, pl.DataFrame):
        df_scores = df_scores.collect()

    return df_scores


def _join_lexicon_from_dataframe(
    df: pl.LazyFrame,
    df_lexicon: pl.LazyFrame,
    df_token_col: str,
    lexicon_token_col: str,
    lexicon_score_col: str,
    impute: float | Literal["mean"] | pl.Expr | None = None,
) -> pl.LazyFrame:
    """ """
    if lexicon_score_col in df.collect_schema().names():
        raise ValueError(f"Column '{lexicon_score_col}' already exists in the dataframe.")

    df_scores = df.join(df_lexicon, left_on=df_token_col, right_on=lexicon_token_col, how="left")

    if impute is not None:
        impute_expr = _get_imputation_expression(
            df_lexicon, lexicon_score_col=lexicon_score_col, impute=impute
        )

        df_scores = df_scores.select(
            pl.all().exclude(lexicon_score_col),
            col(lexicon_score_col).fill_null(impute_expr),
        )

    return df_scores


def _join_lexicon_from_mapping(
    df: pl.LazyFrame,
    lexicon: Mapping[str, float],
    df_token_col: str,
    lexicon_score_alias: str,
    impute: float | Literal["mean"] | pl.Expr | None = None,
) -> pl.LazyFrame:
    """ """
    df_lexicon = pl.LazyFrame(
        list(lexicon.items()), schema=[df_token_col, lexicon_score_alias], orient="row"
    )

    return _join_lexicon_from_dataframe(
        df=df,
        df_lexicon=df_lexicon,
        df_token_col=df_token_col,
        lexicon_token_col=df_token_col,
        lexicon_score_col=lexicon_score_alias,
        impute=impute,
    )


def _get_imputation_expression(
    df_lexicon: pl.LazyFrame, lexicon_score_col: str, impute: float | Literal["mean"] | pl.Expr = 0
) -> pl.Expr:
    """ """
    if isinstance(impute, pl.Expr):
        return impute
    elif isinstance(impute, (int, float)) and not isinstance(impute, bool):
        return pl.lit(impute)
    elif impute == "mean":
        return pl.lit(df_lexicon.select(col(lexicon_score_col).mean()).collect().item())
    else:
        raise ValueError("Invalid input provided to `impute`")
