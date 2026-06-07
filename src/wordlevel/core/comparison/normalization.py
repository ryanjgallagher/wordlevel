from typing import Literal, Sequence, TypeAlias

import polars as pl
import polars.selectors as cs
from polars import col


NORMALIZE_BY: TypeAlias = Literal["sum_abs", "abs_sum"]
r"""
The strategy for normalizing token contributions, dividing each contribution $\delta_\tau$ by a
single constant:

- `"sum_abs"` divides by the sum of absolute contributions, $\sum_\tau |\delta_\tau|$.
- `"abs_sum"` divides by the absolute sum of contributions, $|\sum_\tau \delta_\tau|$.
"""


def normalization_constant_expr(by: NORMALIZE_BY, score_col: str) -> pl.Expr:
    """
    Builds the expression for the scalar by which token contributions are divided to normalize them.

    The two strategies differ in how the per-token contributions $\\delta_\\tau$ in `score_col` are
    aggregated:

    - `"sum_abs"` divides by the sum of absolute contributions, $\\sum_\\tau |\\delta_\\tau|$.
    - `"abs_sum"` divides by the absolute sum of contributions, $|\\sum_\\tau \\delta_\\tau|$.

    The result is a single scalar (broadcast across all rows when used in `with_columns`).
    """
    contributions = col(score_col)
    constant: pl.Expr
    if by == "sum_abs":
        constant = contributions.abs().sum()
    else:
        constant = contributions.sum().abs()

    return constant


def normalize_scores(
    df: pl.DataFrame | pl.LazyFrame,
    by: NORMALIZE_BY,
    score_col: str,
    component_cols: Sequence[str],
    constant_alias: str,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Adds a column with the normalization constant and rescales the token contributions by it.

    Two columns are written: `constant_alias` holds the normalization constant, and `score_col` is
    overwritten with the normalized total contribution. The plot-component columns in
    `component_cols` are rescaled by the same constant, so the decomposition stays coherent: the
    normalized components still sum to the normalized total. Any `component_cols` absent from `df`
    are ignored, so the same call serves both basic plans (only `score_col`) and weighted-average
    plans (which also have the shift-bar components).
    """
    return df.with_columns(
        normalization_constant_expr(by, score_col).alias(constant_alias)
    ).with_columns(cs.by_name(score_col, *component_cols, require_all=False) / col(constant_alias))
