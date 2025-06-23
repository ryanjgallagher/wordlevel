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
    weight1: float, 
    weight2: float,
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
