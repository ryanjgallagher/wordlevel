from typing import Collection

import polars as pl
from polars import col


def exclude_tokens(
    df: pl.DataFrame | pl.LazyFrame,
    token_col: str,
    tokens: Collection[str],
) -> pl.DataFrame | pl.LazyFrame:
    """
    Removes the given tokens from the two corpora being compared.
    """
    return df.filter(~col(token_col).is_in(set(tokens)))
