import polars as pl
import polars.selectors as cs
from polars import col

from wordlevel.constants import (
    ATTRIBUTION_COL,
    PLOT_CONTRIBUTION_COL,
    PLOT_CONTRIBUTION_TYPE_COL,
    RANK_COL,
    SCORE_COL,
    SHIFT_SCORE_TYPES,
    COMPARISON_ATTRIBUTION,
    REFERENCE_ATTRIBUTION,
)


def bar_plot_df(
    df: pl.LazyFrame,
    label_col: str,
    max_rank: int = 50,
    is_nonnegative: bool = False,
    split_nonnegative_by_corpus: bool = True,
) -> pl.DataFrame:
    """ """
    lf_plot = df.filter(col(RANK_COL) <= max_rank)

    # Handle non-negative scores and splitting by corpus
    ref_score = col(SCORE_COL)
    cmp_score = col(SCORE_COL)
    if is_nonnegative and split_nonnegative_by_corpus:
        ref_score = -1 * ref_score

    # Attribute contributions to each corpus
    lf_plot = lf_plot.with_columns(
        pl.when(col(ATTRIBUTION_COL) == REFERENCE_ATTRIBUTION)
        .then(ref_score)
        .otherwise(0)
        .alias(REFERENCE_ATTRIBUTION),
        pl.when(col(ATTRIBUTION_COL) == COMPARISON_ATTRIBUTION)
        .then(cmp_score)
        .otherwise(0)
        .alias(COMPARISON_ATTRIBUTION),
    )

    # Unpivot
    lf_plot = lf_plot.unpivot(
        on=[REFERENCE_ATTRIBUTION, COMPARISON_ATTRIBUTION],
        index=[label_col, RANK_COL],
        value_name=PLOT_CONTRIBUTION_COL,
        variable_name=PLOT_CONTRIBUTION_TYPE_COL,
    )

    return lf_plot.collect()


def _get_shift_plot_df(
    df: pl.LazyFrame,
    label_col: str,
    max_rank: int = 50,
    is_nonnegative: bool = False,
    split_nonnegative_by_corpus: bool = True,
) -> pl.DataFrame:
    """ """
    lf_plot = df.filter(col(RANK_COL) <= max_rank)

    # Handle non-negative scores and splitting by corpus
    if is_nonnegative and split_nonnegative_by_corpus:
        lf_plot = lf_plot.with_columns(
            pl.when(col(ATTRIBUTION_COL) == REFERENCE_ATTRIBUTION)
            .then(-1 * cs.by_name(SHIFT_SCORE_TYPES))
            .otherwise(cs.by_name(SHIFT_SCORE_TYPES))
        )

    # Unpviot
    lf_plot = lf_plot.unpivot(
        on=SHIFT_SCORE_TYPES,
        index=[label_col, RANK_COL],
        value_name=PLOT_CONTRIBUTION_COL,
        variable_name=PLOT_CONTRIBUTION_TYPE_COL,
    )

    return lf_plot.collect()
