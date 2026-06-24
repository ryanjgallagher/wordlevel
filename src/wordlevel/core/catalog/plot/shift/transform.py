import polars as pl
import polars.selectors as cs
from polars import col

from wordlevel.constants import SHIFT_SCORE_TYPES


def bar_plot_df(
    lf: pl.LazyFrame,
    token_col: str,
    score_col: str,
    rank_col: str,
    attribution_col: str,
    contribution_alias: str,
    contribution_type_alias: str,
    attribution_labels: tuple[str, str],
    max_rank: int | None = 50,
    is_nonnegative: bool = False,
    split_nonnegative_by_corpus: bool = True,
    scale: float = 1.0,
) -> pl.DataFrame:
    """
    Builds the plot-ready dataframe for a bar plot.

    `attribution_labels` is the pair of attribution buckets to split contributions into. When a
    non-negative measure is split by corpus, the first bucket's contributions are directed to the
    opposing side of the axis; otherwise each bar follows the sign of its own contribution. Every
    contribution is multiplied by `scale` (e.g. `100` to display normalized scores as percentages).
    
    Passing `max_rank=None` keeps every token (no rank filter)
    """
    first_label, second_label = attribution_labels

    first_score = col(score_col)
    second_score = col(score_col)
    if is_nonnegative and split_nonnegative_by_corpus:
        first_score = -1 * first_score

    if max_rank is not None:
        lf = lf.filter(col(rank_col) <= max_rank)

    lf_plot = (
        lf.with_columns(
            pl.when(col(attribution_col) == first_label)
            .then(first_score)
            .otherwise(0)
            .alias(first_label),
            pl.when(col(attribution_col) == second_label)
            .then(second_score)
            .otherwise(0)
            .alias(second_label),
        )
        .unpivot(
            on=[first_label, second_label],
            index=[token_col, rank_col],
            value_name=contribution_alias,
            variable_name=contribution_type_alias,
        )
        .with_columns(col(contribution_alias) * scale)
    )

    return lf_plot.collect()


def shift_plot_df(
    df: pl.LazyFrame,
    token_col: str,
    rank_col: str,
    attribution_col: str,
    contribution_alias: str,
    contribution_type_alias: str,
    reference_attribution: str = "reference",
    max_rank: int | None = 50,
    is_nonnegative: bool = False,
    split_nonnegative_by_corpus: bool = True,
    scale: float = 1.0,
) -> pl.DataFrame:
    """
    Every contribution is multiplied by `scale` (e.g. `100` to display normalized scores as
    percentages). 
    
    Passing `max_rank=None` keeps every token (no rank filter)
    """
    if max_rank is not None:
        df = df.filter(col(rank_col) <= max_rank)
    
    lf_plot = df

    if is_nonnegative and split_nonnegative_by_corpus:
        lf_plot = lf_plot.with_columns(
            pl.when(col(attribution_col) == reference_attribution)
            .then(cs.by_name(SHIFT_SCORE_TYPES) * pl.lit(-1))
            .otherwise(cs.by_name(SHIFT_SCORE_TYPES))
        )

    lf_plot = lf_plot.unpivot(
        on=SHIFT_SCORE_TYPES,
        index=[token_col, rank_col],
        value_name=contribution_alias,
        variable_name=contribution_type_alias,
    ).with_columns(col(contribution_alias) * scale)

    return lf_plot.collect()
