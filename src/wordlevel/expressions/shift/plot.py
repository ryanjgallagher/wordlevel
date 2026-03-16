import polars as pl
import polars.selectors as cs
from polars import col

from wordlevel.constants import SHIFT_SCORE_TYPES


def bar_plot_df(
    lf: pl.LazyFrame,
    label_col: str,
    score_col: str,
    rank_col: str,
    attribution_col: str,
    contribution_alias: str,
    contribution_type_alias: str,
    reference_attribution: str = "reference",
    comparison_attribution: str = "comparison",
    max_rank: int = 50,
    is_nonnegative: bool = False,
    split_nonnegative_by_corpus: bool = True,
) -> pl.DataFrame:
    """ """
    ref_score = col(score_col)
    cmp_score = col(score_col)
    if is_nonnegative and split_nonnegative_by_corpus:
        ref_score = -1 * ref_score

    lf_plot = (
        lf.filter(col(rank_col) <= max_rank)
        .with_columns(
            pl.when(col(attribution_col) == reference_attribution)
            .then(ref_score)
            .otherwise(0)
            .alias(reference_attribution),
            pl.when(col(attribution_col) == comparison_attribution)
            .then(cmp_score)
            .otherwise(0)
            .alias(comparison_attribution),
        )
        .unpivot(
            on=[reference_attribution, comparison_attribution],
            index=[label_col, rank_col],
            value_name=contribution_alias,
            variable_name=contribution_type_alias,
        )
    )

    return lf_plot.collect()


def shift_plot_df(
    df: pl.LazyFrame,
    label_col: str,
    rank_col: str,
    attribution_col: str,
    contribution_alias: str,
    contribution_type_alias: str,
    reference_attribution: str = "reference",
    max_rank: int = 50,
    is_nonnegative: bool = False,
    split_nonnegative_by_corpus: bool = True,
) -> pl.DataFrame:
    """ """
    lf_plot = df.filter(col(rank_col) <= max_rank)

    if is_nonnegative and split_nonnegative_by_corpus:
        lf_plot = lf_plot.with_columns(
            pl.when(col(attribution_col) == reference_attribution)
            .then(cs.by_name(SHIFT_SCORE_TYPES) * pl.lit(-1))
            .otherwise(cs.by_name(SHIFT_SCORE_TYPES))
        )

    lf_plot = lf_plot.unpivot(
        on=SHIFT_SCORE_TYPES,
        index=[label_col, rank_col],
        value_name=contribution_alias,
        variable_name=contribution_type_alias,
    )

    return lf_plot.collect()
