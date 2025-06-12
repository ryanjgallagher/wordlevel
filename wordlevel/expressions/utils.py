import polars as pl


def normalize_freqs(freqs: pl.Expr, normalize: bool=True) -> pl.Expr:
    """
    """
    if normalize:
        probs = freqs / freqs.sum()
    else:
        probs = freqs

    return probs