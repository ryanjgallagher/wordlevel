from typing import Tuple

import polars as pl


def normalize_freqs(freqs: pl.Expr, normalize: bool=True) -> pl.Expr:
    """
    """
    if normalize:
        probs = freqs / freqs.sum()
    else:
        probs = freqs

    return probs

def mixed_distribution(
    freqs1: pl.Expr, 
    freqs2: pl.Expr, 
    weight1: float=0.5, 
    weight2: float=0.5,
    normalize: bool=True
) -> pl.Expr:
    """
    """
    probs1 = normalize_freqs(freqs1, normalize=normalize)
    probs2 = normalize_freqs(freqs2, normalize=normalize)

    if weight1 + weight2 != 1:
        weight1,weight2 = weight1 / (weight1 + weight2), weight2 / (weight1 + weight2)

    probs_mixed = weight1 * probs1 + weight2 * probs2 

    return probs_mixed

def are_same_sign(expr1: pl.Expr, expr2: pl.Expr) -> pl.Expr:
    """
    """
    same_signs = (
        pl.when(expr1.sign() == expr2.sign()).then(True)
        .when((expr1 == 0) | (expr2 == 0)).then(True)
        .otherwise(False)
    )

    return same_signs

def alias_expressions(
    expr1: pl.Expr, 
    expr2: pl.Expr, 
    alias1: str | None=None, 
    alias2: str | None=None
) -> Tuple[pl.Expr, pl.Expr]:
    """
    """
    if alias1 is not None:
        expr1 = expr1.alias(alias1)

    if alias2 is not None:
        expr2 = expr2.alias(alias2)

    return expr1, expr2