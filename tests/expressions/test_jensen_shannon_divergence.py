import pytest
import polars as pl
import polars.testing

from math import log10

from wordlevel.expressions.entropy import tsallis_entropy
from wordlevel.expressions.jensen_shannon_divergence import (
    jensen_shannon_divergence_shift_scores,
    jensen_shannon_divergence,
    mixed_distribution,
)
from wordlevel.expressions.kullback_leibler_divergence import kullback_leibler_divergence


jsd_params = "freqs1, freqs2, alpha, base, weight1, weight2, expected_scores1, expected_scores2"


@pytest.mark.parametrize(
    jsd_params,
    [
        # Equally mixed distributions, alpha=1, base=2
        (
            [1 / 8, 3 / 8, 2 / 8, 2 / 8],
            [2 / 10, 3 / 10, 1 / 10, 4 / 10],
            1,
            10,
            1 / 2,
            1 / 2,
            [
                (1 / 2) * log10(13 / 10),
                (1 / 2) * log10(9 / 10),
                (1 / 2) * log10(7 / 10),
                (1 / 2) * log10(13 / 10),
            ],
            [
                (1 / 2) * log10(16 / 13),
                (1 / 2) * log10(8 / 9),
                (1 / 2) * log10(4 / 7),
                (1 / 2) * log10(16 / 13),
            ],
        ),
        # Unequally mixed distributions, alpha=3,
        (
            [1 / 8, 3 / 8, 2 / 8, 2 / 8],
            [2 / 10, 3 / 10, 1 / 10, 4 / 10],
            3,
            10,
            1 / 4,
            3 / 4,
            [441 / 204800, -999 / 204800, -279 / 51200, 441 / 51200],
            [549 / 204800, -891 / 204800, -171 / 51200, 549 / 51200],
        ),
        # Unequally mixed distributions, alpha=1/3,
        (
            [1 / 8, 3 / 8, 2 / 8, 2 / 8],
            [2 / 10, 3 / 10, 1 / 10, 4 / 10],
            1 / 3,
            10,
            1,
            2,
            [
                (1 / 3) * ((7 / 40) ** (-2 / 3) - (1 / 8) ** (-2 / 3)) / (-2 / 3),
                (1 / 3) * ((13 / 40) ** (-2 / 3) - (3 / 8) ** (-2 / 3)) / (-2 / 3),
                (1 / 3) * ((6 / 40) ** (-2 / 3) - (2 / 8) ** (-2 / 3)) / (-2 / 3),
                (1 / 3) * ((14 / 40) ** (-2 / 3) - (2 / 8) ** (-2 / 3)) / (-2 / 3),
            ],
            [
                (2 / 3) * ((2 / 10) ** (-2 / 3) - (7 / 40) ** (-2 / 3)) / (-2 / 3),
                (2 / 3) * ((3 / 10) ** (-2 / 3) - (13 / 40) ** (-2 / 3)) / (-2 / 3),
                (2 / 3) * ((1 / 10) ** (-2 / 3) - (6 / 40) ** (-2 / 3)) / (-2 / 3),
                (2 / 3) * ((4 / 10) ** (-2 / 3) - (14 / 40) ** (-2 / 3)) / (-2 / 3),
            ],
        ),
    ],
)
def test_jsd_shift_scores(
    freqs1: list[int],
    freqs2: list[int],
    alpha: float,
    base: float,
    weight1: float,
    weight2: float,
    expected_scores1: list[int],
    expected_scores2: list[int],
) -> None:
    """ """
    df = pl.DataFrame(
        {
            "freqs1": freqs1,
            "freqs2": freqs2,
            "expected_scores1": expected_scores1,
            "expected_scores2": expected_scores2,
        },
        strict=False,
    )
    df = df.with_columns(
        *jensen_shannon_divergence_shift_scores(
            pl.col("freqs1"),
            pl.col("freqs2"),
            base=base,
            alpha=alpha,
            weight1=weight1,
            weight2=weight2,
            alias1="scores1",
            alias2="scores2",
        ),
    )

    polars.testing.assert_series_equal(df["scores1"], df["expected_scores1"], check_names=False)
    polars.testing.assert_series_equal(df["scores2"], df["expected_scores2"], check_names=False)


@pytest.mark.parametrize(
    "freqs1, freqs2, weight1, weight2, base, alpha, expected_total",
    [
        # Identical distributions (expect no divergence)
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1 / 2, 1 / 2, 2, 1, 0),
        # Disjoint distributions (expect complete divergence)
        ([1, 2, 3, 0, 0, 0], [0, 0, 0, 1, 2, 3], 1 / 2, 1 / 2, 2, 1, 1),
        # Distributions varying base, zeros in distributions
        ([1, 2, 1, 0, 2, 1, 2, 0], [1, 1, 0, 2, 2, 1, 1, 1], 1 / 2, 1 / 2, 10, 1, None),
        # Distributions varying alpha and mixed distribution weights
        ([1, 2, 3, 4, 5], [5, 2, 2, 1, 1], 1 / 3, 2 / 3, 2, 2, None),
        # Distributions varying alpha and mixed distribution weights, zeros in distributions
        ([0, 4, 4, 3, 3, 2, 1], [1, 1, 1, 2, 2, 3, 0], 1 / 10, 9 / 10, 2, 1 / 3, None),
    ],
)
def test_jensen_shannon_divergence(
    freqs1: list[int],
    freqs2: list[int],
    weight1: float,
    weight2: float,
    base: float,
    alpha: float,
    expected_total: float,
):
    """
    Test that the JSD is consistent with other ways of computing it:
    JSD = weight_P * KLD(P | M) + weight_Q * KLD(Q | M)
    JSD = H(M) - weight_P * H(P) - weight_Q * H(Q), where H is the Tsallis entropy
    """
    df = pl.DataFrame({"freqs1": freqs1, "freqs2": freqs2}, strict=False)
    df = (
        df.with_columns(
            # Get JSD, mixed distribution, and Tsallis entropy of each individual distribution
            tsallis_entropy(pl.col("freqs1"), alpha=alpha, base=base).alias("entropies1"),
            tsallis_entropy(pl.col("freqs2"), alpha=alpha, base=base).alias("entropies2"),
            jensen_shannon_divergence(
                pl.col("freqs1"),
                pl.col("freqs2"),
                weight1=weight1,
                weight2=weight2,
                base=base,
                alpha=alpha,
            ).alias("jsd"),
            mixed_distribution(
                pl.col("freqs1"), pl.col("freqs2"), weight1=weight1, weight2=weight2
            ).alias("mixed"),
        )
        .with_columns(
            # Use mixed dist to get KLD with respect to it, and Tsallis entropy of mixed dist
            tsallis_entropy(pl.col("mixed"), base=base, alpha=alpha).alias("entropies_mixed"),
            kullback_leibler_divergence(pl.col("freqs1"), pl.col("mixed"), base=base).alias("kld1"),
            kullback_leibler_divergence(pl.col("freqs2"), pl.col("mixed"), base=base).alias("kld2"),
        )
        .with_columns(
            # Calculate JSD via KLD and Tsallis entropy formulas
            (weight1 * pl.col("kld1") + weight2 * pl.col("kld2")).alias("jsd_via_kld"),
            (
                pl.col("entropies_mixed")
                - weight1 * pl.col("entropies1")
                - weight2 * pl.col("entropies2")
            ).alias("jsd_via_entropy"),
        )
    )

    polars.testing.assert_series_equal(df["jsd"], df["jsd_via_entropy"], check_names=False)

    if alpha == 1:
        polars.testing.assert_series_equal(df["jsd"], df["jsd_via_kld"], check_names=False)

    if expected_total is not None:
        assert df["jsd"].sum() == expected_total
