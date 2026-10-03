from collections.abc import Mapping
from typing import Literal, TypeAlias

import polars as pl
from polars import col

from wordlevel.lexicons.core.load import Lexicon


DEFAULT_LEXICON_SCORE_COL = "lexicon_score"

ScoredInOneLexicon: TypeAlias = Literal["borrow", "drop", "mean"]
"""
How to handle a token that is scored by only one of the reference or comparison lexicons.

- `"borrow"` applies the score from the other corpus's lexicon.
- `"drop"` removes the token from the comparison.
- `"mean"` imputes the mean score of the corpus's own lexicon.
"""


def join_lexicon(
    df: pl.DataFrame | pl.LazyFrame,
    lexicon: Lexicon | Mapping[str, float] | pl.DataFrame | pl.LazyFrame,
    df_token_col: str,
    lexicon_token_col: str | None = None,
    lexicon_score_col: str | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """ """
    if isinstance(lexicon, Lexicon):
        df_scores = _join_lexicon_from_dataframe(
            df=df.lazy(),
            df_lexicon=lexicon.lf,
            df_token_col=df_token_col,
            lexicon_token_col=lexicon.token_col,
            lexicon_score_col=lexicon.score_col,
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
        )
    elif isinstance(lexicon, Mapping):
        if lexicon_score_col is None:
            lexicon_score_col = DEFAULT_LEXICON_SCORE_COL

        df_scores = _join_lexicon_from_mapping(
            df=df.lazy(),
            lexicon=lexicon,
            df_token_col=df_token_col,
            lexicon_score_alias=lexicon_score_col,
        )
    else:
        raise TypeError(
            f"The lexicon must be a Mapping or Polars DataFrame/LazyFrame, got: {type(lexicon)}"
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
) -> pl.LazyFrame:
    """ """
    if lexicon_score_col in df.collect_schema().names():
        raise ValueError(f"Column '{lexicon_score_col}' already exists in the dataframe.")

    return df.join(df_lexicon, left_on=df_token_col, right_on=lexicon_token_col, how="left")


def _join_lexicon_from_mapping(
    df: pl.LazyFrame,
    lexicon: Mapping[str, float],
    df_token_col: str,
    lexicon_score_alias: str,
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
    )


def get_imputation_expression(
    lexicon: Lexicon | Mapping[str, float] | pl.DataFrame | pl.LazyFrame,
    impute: float | Literal["mean"] | pl.Expr,
    lexicon_score_col: str | None = None,
) -> pl.Expr:
    """ """
    if isinstance(impute, pl.Expr):
        return impute
    elif isinstance(impute, (int, float)) and not isinstance(impute, bool):
        return pl.lit(impute)
    elif impute != "mean":
        raise ValueError("Invalid input provided to `impute`")

    if isinstance(lexicon, Lexicon):
        return pl.lit(lexicon.lf.select(col(lexicon.score_col).mean()).collect().item())
    elif isinstance(lexicon, (pl.DataFrame, pl.LazyFrame)):
        if lexicon_score_col is None:
            raise ValueError("Must specify `lexicon_score_col` if the lexicon is a dataframe")
        return pl.lit(lexicon.lazy().select(col(lexicon_score_col).mean()).collect().item())
    elif isinstance(lexicon, Mapping):
        return pl.lit(pl.Series(list(lexicon.values())).mean())
    else:
        raise TypeError(
            f"The lexicon must be a Mapping or Polars DataFrame/LazyFrame, got: {type(lexicon)}"
        )
