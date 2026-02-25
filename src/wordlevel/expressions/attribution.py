import polars as pl

from wordlevel.expressions.utils import normalize_freqs


REFERENCE_ATTRIBUTION = "reference"
COMPARISON_ATTRIBUTION = "comparison"
ATTRIBUTIONS = pl.Enum([REFERENCE_ATTRIBUTION, COMPARISON_ATTRIBUTION])


def frequency_attribution(
    reference: pl.Expr, comparison: pl.Expr, normalize: bool = True
) -> pl.Expr:
    """ """
    if normalize:
        reference = normalize_freqs(reference, normalize=normalize)
        comparison = normalize_freqs(comparison, normalize=normalize)

    return (
        pl.when(comparison >= reference)
        .then(pl.lit(COMPARISON_ATTRIBUTION))
        .otherwise(pl.lit(REFERENCE_ATTRIBUTION))
        .cast(ATTRIBUTIONS)
    )


def rank_attribution(reference: pl.Expr, comparison: pl.Expr) -> pl.Expr:
    """ """
    reference_rank = reference.rank(descending=True)
    comparison_rank = comparison.rank(descending=True)

    return (
        pl.when(comparison_rank <= reference_rank)
        .then(pl.lit(COMPARISON_ATTRIBUTION))
        .otherwise(pl.lit(REFERENCE_ATTRIBUTION))
        .cast(ATTRIBUTIONS)
    )
