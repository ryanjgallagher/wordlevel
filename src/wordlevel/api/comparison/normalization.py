from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from wordlevel.constants import (
    SCORE_COL,
    SHIFT_SCORE_TYPES,
    NORMALIZATION_CONSTANT_COL,
)
from wordlevel.core.comparison.normalization import (
    NORMALIZE_BY,
    normalization_constant_expr,
    normalize_scores,
)

if TYPE_CHECKING:
    from wordlevel.api.comparison.state import PlanStage


_ZERO_CONSTANT_DESCRIPTION: dict[NORMALIZE_BY, str] = {
    "sum_abs": "the total absolute contribution is zero (the corpora are identical)",
    "abs_sum": "the net difference is zero",
}


def make_normalized_plan(by: NORMALIZE_BY) -> PlanStage:
    """
    Creates a plan that normalizes token contributions by a single scalar.

    The plan collects the scored contributions once, then validates that the normalization constant
    is non-zero — a divide-by-zero that depends on the data, so it can only be caught once the
    scores are realized. If the constant is zero, it raises; otherwise it normalizes the scores.
    """

    def plan(
        df: pl.DataFrame | pl.LazyFrame, token_col: str, normalization_suffix: str
    ) -> pl.DataFrame | pl.LazyFrame:
        df_scored = df.lazy().collect()

        constant = df_scored.select(normalization_constant_expr(by, SCORE_COL)).item()
        if constant == 0:
            raise ValueError(
                f"Cannot normalize by {by!r}: {_ZERO_CONSTANT_DESCRIPTION[by]}, so the "
                f"contributions cannot be rescaled."
            )

        return normalize_scores(
            df_scored.lazy(),
            by=by,
            score_col=SCORE_COL,
            component_cols=SHIFT_SCORE_TYPES,
            constant_alias=NORMALIZATION_CONSTANT_COL,
        )

    return plan
