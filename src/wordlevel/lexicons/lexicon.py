from typing import Literal, Mapping

import polars as pl
from polars import col

from wordlevel.types import PolarsFrame


def join_lexicon(
    df: PolarsFrame,
    lexicon: Mapping[str, float] | PolarsFrame,
    df_label_col: str,
    lexicon_label_col: str | None = None,
    lexicon_score_col: str | None = None,
    impute: float | Literal["mean"] | pl.Expr = 0,
) -> PolarsFrame:
    """ """
    if isinstance(lexicon, PolarsFrame):
        if lexicon_label_col is None:
            raise ValueError("Must specify the `lexicon_label_col` if the lexicon is a dataframe")
        if lexicon_score_col is None:
            raise ValueError("Must specify the `lexicon_score_col` if the lexicon is a dataframe")

        df_lexicon = lexicon.lazy()
    elif isinstance(lexicon, Mapping):
        if lexicon_label_col is None:
            lexicon_label_col = df_label_col
        if lexicon_score_col is None:
            lexicon_score_col = "score"

        df_lexicon = pl.LazyFrame(
            list(lexicon.items()), schema=[lexicon_label_col, lexicon_score_col], orient="row"
        )
    else:
        raise ValueError(f"Invalid input provided to `lexicon` of type: {type(lexicon)}")

    if isinstance(impute, (int, float)):
        impute_expr = pl.lit(impute)
    elif isinstance(impute, pl.Expr):
        impute_expr = impute
    elif impute == "mean":
        impute_expr = pl.lit(df_lexicon.select(col(lexicon_score_col).mean()).collect().item())
    else:
        raise ValueError("Invalid input provided to `impute`")

    df_lexicon_score = (
        df.lazy()
        .join(df_lexicon, left_on=df_label_col, right_on=lexicon_label_col, how="left")
        .select(df.columns, col(lexicon_score_col).fill_null(impute_expr))
    )
    if isinstance(df, pl.DataFrame):
        df_lexicon_score = df_lexicon_score.collect()

    return df_lexicon_score
