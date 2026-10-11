from collections.abc import Collection
from typing import Literal, TypeAlias

import polars as pl
from polars import col

from wordlevel.constants import FREQ_CMP_COL, FREQ_REF_COL


ExcludeCorpus: TypeAlias = Literal["reference", "comparison", "both", "either"]
"""
Which corpus or corpora are used to determine whether a token should be excluded.

- `"reference"` excludes a token when its value in the reference corpus falls in the range.
- `"comparison"` excludes a token when its value in the comparison corpus falls in the range.
- `"both"` excludes a token only when its value falls in the range in both corpora.
- `"either"` excludes a token when its value falls in the range in at least one corpus.
"""

RankTies: TypeAlias = Literal["average", "min", "max", "dense"]
"""
How tokens sharing a frequency are ranked.

Tokens of equal frequency form a tie block, and every method keeps that block together at a single
shared rank. In the frequencies `[10, 5, 5, 5, 1]`, the three 5s form one tie block.

- `"average"` gives the block the mean of the ranks it spans. In the example, this produces rank 3.
- `"min"` gives the block the first of the ranks it spans. In the example, this produces rank 2.
- `"max"` gives the block the last of the ranks it spans. In the example, this produces rank 4.
- `"dense"` numbers the distinct frequencies rather than the tokens, so a tie block takes up one
  rank instead of as many ranks as it has tokens. In the example, this produces rank 2 for the
  block and rank 3 for the 1, where the other three methods leave the 1 at rank 5.
"""

FREQ_RANK_SUFFIX = "_rank_{ties}"


def freq_rank_col(freq_col: str, ties: RankTies) -> str:
    """
    Derives the name of the column holding the frequency ranks of one corpus, given the name of the
    column holding its frequencies.
    """
    return f"{freq_col}{FREQ_RANK_SUFFIX.format(ties=ties)}"


def rank_freqs(
    df: pl.DataFrame | pl.LazyFrame,
    ties: Collection[RankTies],
) -> pl.DataFrame | pl.LazyFrame:
    """
    Adds columns representing the frequency rank of each token in the two corpora being compared.

    Rank 1 is the most common token of a corpus, and a token a corpus does not use has a frequency
    of zero in it, placing it in that corpus's bottom tie block. One pair of columns is added per
    tie method, so that a comparison can exclude by rank more than once with different tie
    handling. The columns are added in the order `ties` gives.
    """
    rank_cols: list[pl.Expr] = []
    for tie_method in ties:
        for freq_col in (FREQ_REF_COL, FREQ_CMP_COL):
            ranks = col(freq_col).rank(method=tie_method, descending=True)
            rank_cols.append(ranks.alias(freq_rank_col(freq_col, tie_method)))

    return df.with_columns(*rank_cols)


def exclude_tokens(
    df: pl.DataFrame | pl.LazyFrame,
    token_col: str,
    tokens: Collection[str],
) -> pl.DataFrame | pl.LazyFrame:
    """
    Removes the given tokens from the two corpora being compared.
    """
    return df.filter(~col(token_col).is_in(set(tokens)))


def exclude_by_range(
    df: pl.DataFrame | pl.LazyFrame,
    value_col_ref: str,
    value_col_cmp: str,
    less_than: float | None = None,
    greater_than: float | None = None,
    between: tuple[float, float] | None = None,
    inclusive: bool = True,
    corpus: ExcludeCorpus = "both",
) -> pl.DataFrame | pl.LazyFrame:
    """
    Removes tokens whose per-corpus value falls in the given range from the two corpora being
    compared.

    The value columns are whatever the caller wants the range tested against, such as lexicon
    scores or frequencies. Exactly one of `less_than`, `greater_than`, or `between` gives the
    range, and `corpus` decides which corpus's value the range is tested against.
    """
    if (less_than is None) and (greater_than is None) and (between is None):
        raise ValueError("Must specify one of `less_than`, `greater_than`, or `between`")

    condition_ref = _get_range_condition(
        col(value_col_ref), less_than, greater_than, between, inclusive
    )
    condition_cmp = _get_range_condition(
        col(value_col_cmp), less_than, greater_than, between, inclusive
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


def _get_range_condition(
    values: pl.Expr,
    less_than: float | None,
    greater_than: float | None,
    between: tuple[float, float] | None,
    inclusive: bool,
) -> pl.Expr:
    """
    Builds the condition that a value falls in the given range.
    """
    condition: pl.Expr
    if less_than is not None:
        condition = values <= less_than if inclusive else values < less_than
    elif greater_than is not None:
        condition = values >= greater_than if inclusive else values > greater_than
    else:
        lower, upper = between  # type: ignore[misc]
        condition = values.is_between(lower, upper, closed="both" if inclusive else "none")

    return condition
