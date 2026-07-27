import polars as pl
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
    passthrough_cols: list[str] | None = None,
    scale: float = 1.0,
) -> pl.DataFrame:
    """
    Builds the plot-ready dataframe for a bar plot.

    `attribution_labels` is the pair of attribution buckets to split contributions into. When a
    non-negative measure is split by corpus, the first bucket's contributions are directed to the
    opposing side of the axis; otherwise each bar follows the sign of its own contribution. Every
    contribution is multiplied by `scale` (e.g. `100` to display normalized scores as percentages).

    Passing `max_rank=None` keeps every token (no rank filter)

    `passthrough_cols` are carried through the unpivot as additional index columns
    """
    unpivot_index_cols = [token_col, rank_col]
    if passthrough_cols is not None:
        unpivot_index_cols.extend(passthrough_cols)

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
            index=unpivot_index_cols,
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
    contribution_alias: str,
    contribution_type_alias: str,
    max_rank: int | None = 50,
    passthrough_cols: list[str] | None = None,
    scale: float = 1.0,
) -> pl.DataFrame:
    """
    Every contribution is multiplied by `scale` (e.g. `100` to display normalized scores as
    percentages).

    Passing `max_rank=None` keeps every token (no rank filter)

    `passthrough_cols` are carried through the unpivot as additional index columns
    """
    unpivot_index_cols = [token_col, rank_col]
    if passthrough_cols is not None:
        unpivot_index_cols.extend(passthrough_cols)

    if max_rank is not None:
        df = df.filter(col(rank_col) <= max_rank)

    lf_plot = df.unpivot(
        on=SHIFT_SCORE_TYPES,
        index=unpivot_index_cols,
        value_name=contribution_alias,
        variable_name=contribution_type_alias,
    ).with_columns(col(contribution_alias) * scale)

    return lf_plot.collect()
