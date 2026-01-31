import polars as pl


def frequency_difference(freqs1: pl.Expr, freqs2: pl.Expr) -> pl.Expr:
    """Compute the difference in frequencies between two distributions."""
    return freqs2 - freqs1
