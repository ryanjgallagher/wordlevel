import polars as pl


def frequency_difference(freqs1: pl.Expr, freqs2: pl.Expr) -> pl.Expr:
    """
    Computes the difference in each token's frequency between the two corpora. A positive
    difference means the token is used more in the second corpus.
    """
    return freqs2 - freqs1
