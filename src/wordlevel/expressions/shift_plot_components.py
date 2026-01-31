import polars as pl

from wordlevel.types import PolarsFrame
from wordlevel.expressions.utils import are_same_sign


def get_shift_plot_components(
    df: PolarsFrame,
    frequency_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    more_of_positive_alias: str = "more_of_positive",
    less_of_negative_alias: str = "less_of_negative",
    more_of_negative_alias: str = "more_of_negative",
    less_of_positive_alias: str = "less_of_positive",
    higher_score_alias: str = "higher_score",
    lower_score_alias: str = "lower_score",
    counteracted_suffix: str = "_counteracted",
) -> PolarsFrame:
    """ """
    components = {
        "frequency_diff": frequency_diff,
        "weighted_score_diff": weighted_score_diff,
        "reference_score_diff": reference_score_diff,
    }

    # fmt: off
    df_comps = df.with_columns(
        more_of_positive(**components, counteracted=False).alias(more_of_positive_alias),
        less_of_negative(**components, counteracted=False).alias(less_of_negative_alias),
        more_of_negative(**components, counteracted=False).alias(more_of_negative_alias),
        less_of_positive(**components, counteracted=False).alias(less_of_positive_alias),
        higher_score(**components, counteracted=False).alias(higher_score_alias),
        lower_score(**components, counteracted=False).alias(lower_score_alias),
        more_of_positive(**components, counteracted=True).alias(f"{more_of_positive_alias}{counteracted_suffix}"),
        less_of_negative(**components, counteracted=True).alias(f"{less_of_negative_alias}{counteracted_suffix}"),
        more_of_negative(**components, counteracted=True).alias(f"{more_of_negative_alias}{counteracted_suffix}"),
        less_of_positive(**components, counteracted=True).alias(f"{less_of_positive_alias}{counteracted_suffix}"),
        higher_score(**components, counteracted=True).alias(f"{higher_score_alias}{counteracted_suffix}"),
        lower_score(**components, counteracted=True).alias(f"{lower_score_alias}{counteracted_suffix}"),
    )
    # fmt: on

    return df_comps


def plot_component(component1: pl.Expr, component2: pl.Expr) -> pl.Expr:
    """ """
    # fmt: off
    return (
        pl.when(are_same_sign(component1, component2)).then(
            component1
        ).otherwise(
            pl.when(component1.abs() > component2.abs()).then(
                component1 + component2
            ).otherwise(
                0
            )
        )
    )
    # fmt: on


def plot_component_counteracted(component1: pl.Expr, component2: pl.Expr) -> pl.Expr:
    """ """
    # fmt: off
    return (
        pl.when(are_same_sign(component1, component2)).then(
            0
        ).otherwise(
            pl.when(component1.abs() > component2.abs()).then(
                -1 * component2
            ).otherwise(
                component1
            )
        )
    )
    # fmt: on


def conditional_plot_component(
    component1: pl.Expr,
    component2: pl.Expr,
    condition: pl.Expr,
    counteracted: bool,
) -> pl.Expr:
    """ """
    if counteracted:
        plot_component_func = plot_component_counteracted
    else:
        plot_component_func = plot_component

    return pl.when(condition).then(plot_component_func(component1, component2)).otherwise(0)


def more_of_positive(
    frequency_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    counteracted: bool,
) -> pl.Expr:
    """ """
    return conditional_plot_component(
        component1=frequency_diff * reference_score_diff,
        component2=weighted_score_diff,
        condition=(frequency_diff >= 0) & (reference_score_diff >= 0),
        counteracted=counteracted,
    )


def less_of_negative(
    frequency_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    counteracted: bool,
) -> pl.Expr:
    """ """
    return conditional_plot_component(
        component1=frequency_diff * reference_score_diff,
        component2=weighted_score_diff,
        condition=(frequency_diff < 0) & (reference_score_diff < 0),
        counteracted=counteracted,
    )


def more_of_negative(
    frequency_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    counteracted: bool,
) -> pl.Expr:
    """ """
    return conditional_plot_component(
        component1=frequency_diff * reference_score_diff,
        component2=weighted_score_diff,
        condition=(frequency_diff >= 0) & (reference_score_diff < 0),
        counteracted=counteracted,
    )


def less_of_positive(
    frequency_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    counteracted: bool,
) -> pl.Expr:
    """ """
    return conditional_plot_component(
        component1=frequency_diff * reference_score_diff,
        component2=weighted_score_diff,
        condition=(frequency_diff < 0) & (reference_score_diff >= 0),
        counteracted=counteracted,
    )


def higher_score(
    frequency_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    counteracted: bool,
) -> pl.Expr:
    """ """
    return conditional_plot_component(
        component1=weighted_score_diff,
        component2=frequency_diff * reference_score_diff,
        condition=(weighted_score_diff >= 0),
        counteracted=counteracted,
    )


def lower_score(
    frequency_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    counteracted: bool,
) -> pl.Expr:
    """ """
    return conditional_plot_component(
        component1=weighted_score_diff,
        component2=frequency_diff * reference_score_diff,
        condition=(weighted_score_diff < 0),
        counteracted=counteracted,
    )
