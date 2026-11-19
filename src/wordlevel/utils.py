import polars as pl


def normalize_freqs(freqs: pl.Expr, normalize: bool = True) -> pl.Expr:
    """
    Normalizes frequencies to sum to 1. If `normalize` is `False`, returns them unchanged.
    """
    if normalize:
        probs = freqs / freqs.sum()
    else:
        probs = freqs

    return probs


def mixed_distribution(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    weight1: float = 0.5,
    weight2: float = 0.5,
    normalize: bool = True,
) -> pl.Expr:
    r"""
    Computes each token's normalized frequency in a mixture of two corpora:

    $$
    p_\tau^{(M)} = \pi_1 \, p_\tau^{(1)} + \pi_2 \, p_\tau^{(2)}
    $$

    where $p_\tau^{(1)}$ and $p_\tau^{(2)}$ are the token's normalized frequencies in each corpus,
    and $\pi_1$ and $\pi_2$ are `weight1` and `weight2`. If the weights do not sum to 1, they are
    rescaled so that they do. If `normalize` is `False`, `freqs1` and `freqs2` are used as given
    rather than normalized to sum to 1.
    """
    probs1 = normalize_freqs(freqs1, normalize=normalize)
    probs2 = normalize_freqs(freqs2, normalize=normalize)

    if weight1 + weight2 != 1:
        weight1, weight2 = weight1 / (weight1 + weight2), weight2 / (weight1 + weight2)

    probs_mixed = weight1 * probs1 + weight2 * probs2

    return probs_mixed


def are_same_sign(expr1: pl.Expr, expr2: pl.Expr) -> pl.Expr:
    """
    Checks, for each pair of values in `expr1` and `expr2`, whether both are non-negative or both
    are non-positive. So if either value is 0, the check passes.
    """
    same_signs = (
        pl.when(expr1.sign() == expr2.sign())
        .then(True)
        .when((expr1 == 0) | (expr2 == 0))
        .then(True)
        .otherwise(False)
    )

    return same_signs


def alias_expressions(
    expr1: pl.Expr,
    expr2: pl.Expr,
    alias1: str | None = None,
    alias2: str | None = None,
) -> tuple[pl.Expr, pl.Expr]:
    """
    Names a pair of expressions `alias1` and `alias2`. An expression whose alias is `None` keeps
    its current name.
    """
    if alias1 is not None:
        expr1 = expr1.alias(alias1)

    if alias2 is not None:
        expr2 = expr2.alias(alias2)

    return expr1, expr2
