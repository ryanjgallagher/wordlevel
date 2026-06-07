from __future__ import annotations

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
from wordlevel.api.comparison.score import ScorePlan


_ZERO_CONSTANT_DESCRIPTION: dict[NORMALIZE_BY, str] = {
    "sum_abs": "the total absolute contribution is zero (the corpora are identical)",
    "abs_sum": "the net difference is zero",
}


def make_normalized_plan(base_plan: ScorePlan, by: NORMALIZE_BY) -> ScorePlan:
    """
    Wraps a scoring plan so that its token contributions are normalized by a single scalar.

    The wrapped plan runs `base_plan`, then validates that the normalization constant is non-zero —
    a divide-by-zero that depends on the data, so it can only be caught once the scores are realized
    (mirroring the Kullback-Leibler divergence's `validated_plan`). If the constant is zero, it
    raises; otherwise it rescales the contributions via the pure-core `normalize_scores`.
    """

    def plan(
        df: pl.DataFrame | pl.LazyFrame, token_col: str, normalization_suffix: str
    ) -> pl.DataFrame | pl.LazyFrame:
        df_scored = base_plan(df, token_col, normalization_suffix)

        constant = (
            df_scored.lazy()
            .select(normalization_constant_expr(by, SCORE_COL).alias(NORMALIZATION_CONSTANT_COL))
            .collect()
            .item()
        )
        if constant == 0:
            raise ValueError(
                f"Cannot normalize by {by!r}: {_ZERO_CONSTANT_DESCRIPTION[by]}, so the "
                f"contributions cannot be rescaled."
            )

        return normalize_scores(
            df_scored,
            by=by,
            score_col=SCORE_COL,
            component_cols=SHIFT_SCORE_TYPES,
            constant_alias=NORMALIZATION_CONSTANT_COL,
        )

    return plan
