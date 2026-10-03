from math import log10

import polars as pl
import polars.testing
import pytest

from wordlevel.comparison.core.measures.entropy import tsallis_entropy_dist_scores
from wordlevel.comparison.core.measures.kullback_leibler_divergence import (
    kullback_leibler_divergence,
    kullback_leibler_divergence_dual_dist_scores,
)


@pytest.mark.parametrize(
    "freqs1, freqs2, base", [([1, 2, 3, 4], [7, 5, 3, 1], 2), ([1, 1, 1, 1], [1, 2, 2, 1], 10)]
)
def test_kld_shift_scores(freqs1: list[int], freqs2: list[int], base: float) -> None:
    """ """
    df = pl.DataFrame({"freqs1": freqs1, "freqs2": freqs2}, strict=False)
    df = df.with_columns(
        *kullback_leibler_divergence_dual_dist_scores(
            pl.col("freqs1"), pl.col("freqs2"), base=base, alias1="scores1", alias2="scores2"
        ),
        entropies1=tsallis_entropy_dist_scores(pl.col("freqs1"), base=base),
        entropies2=tsallis_entropy_dist_scores(pl.col("freqs2"), base=base),
    )

    # Scores are Tsallis surprisals but positionally swapped relative to (freqs1, freqs2)
    # so the framework's `p_2 * s_2 - p_1 * s_1` computes D_KL(P_2 || P_1) under
    # share_component_freqs_by="comparison". See metric docstring for the rearrangement.
    polars.testing.assert_series_equal(df["scores1"], df["entropies2"], check_names=False)
    polars.testing.assert_series_equal(df["scores2"], df["entropies1"], check_names=False)


@pytest.mark.parametrize(
    "freqs1, freqs2, base, expected",
    [
        # Identical distributions (expect no divergence)
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 2, 0.0),
        # P_2 (data) has a token P_1 (codebook) lacks (expect infinite divergence)
        (
            [5, 4, 3, 2, 0],  # freqs1 = codebook
            [1, 2, 3, 4, 1],  # freqs2 = data
            2,
            float("inf"),
        ),
        # Non-identical distributions, varying base, zeros that shouldn't result in infinite divergence
        (
            [5, 2, 5, 1, 3, 1, 3, 0],  # freqs1 = codebook
            [1, 2, 3, 4, 5, 5, 0, 0],  # freqs2 = data
            10,
            [
                (1 / 20) * log10(1 / 5),
                (2 / 20) * log10(1),
                (3 / 20) * log10(3 / 5),
                (4 / 20) * log10(4),
                (5 / 20) * log10(5 / 3),
                (5 / 20) * log10(5),
                0,
                0,
            ],
        ),
    ],
)
def test_kullback_leibler_divergence(
    freqs1: list[int], freqs2: list[int], base: float, expected: float
):
    """ """
    df = pl.DataFrame({"freqs1": freqs1, "freqs2": freqs2, "expected": expected}, strict=False)
    df = df.with_columns(
        kullback_leibler_divergence(pl.col("freqs1"), pl.col("freqs2"), base=base).alias("kld")
    )

    if isinstance(expected, float):
        assert df["kld"].sum() == expected
    else:
        polars.testing.assert_series_equal(df["kld"], df["expected"], check_names=False)
