import polars as pl 

from wordlevel.expressions.utils import are_same_sign


def plot_component(component1: pl.Expr, component2: pl.Expr) -> pl.Expr:
    """
    """
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

def plot_component_counteracted(component1: pl.Expr, component2: pl.Expr) -> pl.Expr:
    """
    """
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

def conditional_plot_component(
    component1: pl.Expr,
    component2: pl.Expr,
    condition: pl.Expr,
    counteracted: bool
) -> pl.Expr:
    """
    """
    if counteracted:
        plot_component_func = plot_component_counteracted
    else:
        plot_component_func = plot_component

    return pl.when(condition).then(plot_component_func(component1, component2)).otherwise(0)

def more_of_positive(
    frequency_diff: pl.Expr, 
    reference_score_diff: pl.Expr, 
    weighted_score_diff: pl.Expr,
    counteracted: bool
) -> pl.Expr:
    """
    """
    return conditional_plot_component(
        component1=frequency_diff * reference_score_diff, 
        component2=weighted_score_diff, 
        condition=(frequency_diff >= 0) & (reference_score_diff >= 0),
        counteracted=counteracted
    )

def less_of_negative(
    frequency_diff: pl.Expr, 
    reference_score_diff: pl.Expr, 
    weighted_score_diff: pl.Expr,
    counteracted: bool
) -> pl.Expr:
    """
    """
    return conditional_plot_component(
        component1=frequency_diff * reference_score_diff, 
        component2=weighted_score_diff, 
        condition=(frequency_diff < 0) & (reference_score_diff < 0),
        counteracted=counteracted
    )

def more_of_negative(
    frequency_diff: pl.Expr, 
    reference_score_diff: pl.Expr, 
    weighted_score_diff: pl.Expr,
    counteracted: bool
) -> pl.Expr:
    """
    """
    return conditional_plot_component(
        component1=frequency_diff * reference_score_diff, 
        component2=weighted_score_diff, 
        condition=(frequency_diff >= 0) & (reference_score_diff < 0),
        counteracted=counteracted
    )

def less_of_positive(
    frequency_diff: pl.Expr, 
    reference_score_diff: pl.Expr, 
    weighted_score_diff: pl.Expr,
    counteracted: bool
) -> pl.Expr:
    """
    """
    return conditional_plot_component(
        component1=frequency_diff * reference_score_diff, 
        component2=weighted_score_diff, 
        condition=(frequency_diff < 0) & (reference_score_diff >= 0),
        counteracted=counteracted
    )

def higher_score(
    frequency_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    counteracted: bool
) -> pl.Expr:
    """
    """
    return conditional_plot_component(
        component1=weighted_score_diff, 
        component2=frequency_diff * reference_score_diff, 
        condition=(weighted_score_diff >= 0),
        counteracted=counteracted
    )

def lower_score(
    frequency_diff: pl.Expr,
    reference_score_diff: pl.Expr,
    weighted_score_diff: pl.Expr,
    counteracted: bool
) -> pl.Expr:
    """
    """
    return conditional_plot_component(
        component1=weighted_score_diff, 
        component2=frequency_diff * reference_score_diff, 
        condition=(weighted_score_diff < 0),
        counteracted=counteracted
    )