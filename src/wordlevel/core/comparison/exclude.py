from typing import Collection, Literal

import polars as pl
from polars import col


EXCLUDE_CORPUS = Literal["reference", "comparison", "both", "either"]


def exclude_tokens(
    df: pl.DataFrame | pl.LazyFrame,
    token_col: str,
    tokens: Collection[str],
) -> pl.DataFrame | pl.LazyFrame:
    """
    Removes the given tokens from the two corpora being compared.
    """
    return df.filter(~col(token_col).is_in(set(tokens)))


def exclude_by_lexicon_score(
    df: pl.DataFrame | pl.LazyFrame,
    score_col_ref: str,
    score_col_cmp: str,
    less_than: float | None = None,
    greater_than: float | None = None,
    between: tuple[float, float] | None = None,
    inclusive: bool = True,
    corpus: EXCLUDE_CORPUS = "both",
) -> pl.DataFrame | pl.LazyFrame:
    """
    Removes tokens whose lexicon score falls in the given range from the two corpora being
    compared.

    Exactly one of `less_than`, `greater_than`, or `between` gives the range, and `corpus` decides
    which corpus's lexicon score the range is tested against.
    """
    if (less_than is None) and (greater_than is None) and (between is None):
        raise ValueError("Must specify one of `less_than`, `greater_than`, or `between`")

    condition_ref = _get_score_condition(
        col(score_col_ref), less_than, greater_than, between, inclusive
    )
    condition_cmp = _get_score_condition(
        col(score_col_cmp), less_than, greater_than, between, inclusive
    )

    condition: pl.Expr
    if corpus == "reference":
        condition = condition_ref
    elif corpus == "comparison":
        condition = condition_cmp
    elif corpus == "both":
        condition = condition_ref & condition_cmp
    else:
        condition = condition_ref | condition_cmp

    return df.filter(~condition)


def _get_score_condition(
    scores: pl.Expr,
    less_than: float | None,
    greater_than: float | None,
    between: tuple[float, float] | None,
    inclusive: bool,
) -> pl.Expr:
    """
    Builds the condition that a lexicon score falls in the given range.
    """
    condition: pl.Expr
    if less_than is not None:
        condition = scores <= less_than if inclusive else scores < less_than
    elif greater_than is not None:
        condition = scores >= greater_than if inclusive else scores > greater_than
    else:
        lower, upper = between  # type: ignore[misc]
        condition = scores.is_between(lower, upper, closed="both" if inclusive else "none")

    return condition
