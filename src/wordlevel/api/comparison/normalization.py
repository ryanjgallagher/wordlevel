from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from wordlevel.constants import (
    NORMALIZATION_CONSTANT_COL,
    SCORE_COL,
    SHIFT_SCORE_TYPES,
)
from wordlevel.core.comparison.normalization import (
    NormalizeBy,
    normalize_scores,
)


if TYPE_CHECKING:
    from wordlevel.api.comparison.state import PlanStage


_ZERO_CONSTANT_DESCRIPTION: dict[NormalizeBy, str] = {
    "total_diff": "the total difference is zero (the corpora are identical)",
    "net_diff": "the net difference is zero",
}


def make_normalized_plan(by: NormalizeBy, check_nonzero: bool = True) -> PlanStage:
    """
    Creates a plan that normalizes token contributions by a single scalar.

    By default, the plan collects the scored contributions once, then validates that the
    normalization constant is non-zero — a divide-by-zero that depends on the data, so it can only
    be caught once the scores are realized. If the constant is zero, it raises; otherwise it
    normalizes the scores.

    The user can bypass the check for the zero constant by setting `check_nonzero=False`.
    """

    def plan(df: pl.DataFrame | pl.LazyFrame, token_col: str) -> pl.DataFrame | pl.LazyFrame:
        lf_normalized = normalize_scores(
            df.lazy(),
            by=by,
            score_col=SCORE_COL,
            component_cols=SHIFT_SCORE_TYPES,
            constant_alias=NORMALIZATION_CONSTANT_COL,
        ).lazy()

        if not check_nonzero:
            return lf_normalized

        df_normalized = lf_normalized.collect()
        if df_normalized[NORMALIZATION_CONSTANT_COL].first() == 0:
            raise ValueError(
                f"Cannot normalize by {by!r}: {_ZERO_CONSTANT_DESCRIPTION[by]}, so the "
                f"contributions cannot be rescaled."
            )

        return df_normalized.lazy()

    return plan
