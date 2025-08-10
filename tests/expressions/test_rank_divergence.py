import pytest
import polars as pl
import polars.testing

from math import log
from wordlevel.expressions.rank_divergence import *


@pytest.mark.parametrize(
    "input, expected_output, alpha",
    [
        # In-order ranking
        ([1, 2, 3], [3, 2, 1.0], 1),
        # Out-of-order ranking
        ([3, 4, 2, 5, 1], [3, 2, 4, 1.0, 5], 1),
        # Ties
        ([1, 1, 1, 1], [10 / 4, 10 / 4, 10 / 4, 10 / 4], 1),
        # Ties mixed in
        ([3, 2, 2, 1], [1.0, 5 / 2, 5 / 2, 4], 1),
        # alpha > 1
        ([3, 2, 1], [1.0, 2**2, 3**2], 2),
        # alpha < 1
        ([3, 2, 1], [1.0, 2**0.5, 3**0.5], 0.5),
        # Out-of-order, ties, alpha > 1
        (
            [4, 5, 3, 1, 2, 2, 1, 1],
            [
                2**3,
                1.0,
                (3**3),
                (21 / 3) ** 3,
                (9 / 2) ** 3,
                (9 / 2) ** 3,
                (21 / 3) ** 3,
                (21 / 3) ** 3,
            ],
            3,
        ),
    ],
)
def test_rank(input: list[int | float], expected_output: list[float], alpha: float) -> None:
    """ """
    df = pl.DataFrame({"input": input, "expected_output": expected_output}, strict=False)
    df = df.with_columns(actual_output=rank(pl.col("input"), alpha))

    polars.testing.assert_series_equal(
        df["expected_output"], df["actual_output"], check_names=False
    )

    # Check that the ranks sum to N * (N + 1) / 2 if there are N elements
    if alpha == 1:
        n = len(input)
        assert df["actual_output"].sum() == n * (n + 1) / 2


@pytest.mark.parametrize(
    "freqs1, freqs2, expected_output, alpha",
    [
        # Identical systems, alpha=1
        ([3, 2, 1], [3, 2, 1], [0.0, 0.0, 0.0], 1),
        # Disjoint systems, alpha=1
        (
            [2, 1, 0, 0, 0],
            [0, 0, 3, 2, 1],
            [
                2 * (7 / 9) ** (1 / 2),
                2 * (5 / 18) ** (1 / 2),
                2 * (3 / 4) ** (1 / 2),
                2 * (1 / 4) ** (1 / 2),
                2 * (1 / 12) ** (1 / 2),
            ],
            1,
        ),
        # Disjoint systems, alpha=2
        (
            [2, 1, 0, 0, 0],
            [0, 0, 3, 2, 1],
            [
                (3 / 2) * (77 / 81) ** (1 / 3),
                (3 / 2) * (65 / 324) ** (1 / 3),
                (3 / 2) * (15 / 16) ** (1 / 3),
                (3 / 2) * (3 / 16) ** (1 / 3),
                (3 / 2) * (7 / 144) ** (1 / 3),
            ],
            2,
        ),
        # Overlapping systems, alpha=1/3
        (
            [2, 1, 0, 0],
            [0, 3, 2, 1],
            [
                4 * abs(1 - (1 / 4) ** (1 / 3)) ** (3 / 4),
                4 * abs((1 / 2) ** (1 / 3) - 1) ** (3 / 4),
                4 * abs((2 / 7) ** (1 / 3) - (1 / 2) ** (1 / 3)) ** (3 / 4),
                4 * abs((2 / 7) ** (1 / 3) - (1 / 3) ** (1 / 3)) ** (3 / 4),
            ],
            1 / 3,
        ),
        # Overlapping systems, alpha=0
        (
            [5, 3, 2, 1, 0, 0],
            [5, 0, 4, 3, 2, 1],
            [
                0.0,
                abs(log(2) - log(6)),
                abs(log(3) - log(2)),
                abs(log(4) - log(3)),
                abs(log(11 / 2) - log(4)),
                abs(log(11 / 2) - log(5)),
            ],
            0,
        ),
        # Overlapping systems, alpha=inf
        (
            [5, 3, 2, 1, 0, 0],
            [5, 0, 4, 3, 2, 1],
            [0.0, 1 / 2, 1 / 2, 1 / 3, 1 / 4, 1 / 5],
            float("inf"),
        ),
    ],
)
def test_rank_divergence_summands(freqs1: list[int], freqs2: list[int], expected_output: list[float], alpha: float) -> None:
    """ """
    df = pl.DataFrame({"freqs1": freqs1, "freqs2": freqs2, "expected_output": expected_output})
    df = df.with_columns(
        actual_output=rank_divergence_summands(pl.col("freqs1"), pl.col("freqs2"), alpha=alpha)
    )

    with pl.Config(tbl_width_chars=2000):
        print(df)

    polars.testing.assert_series_equal(
        df["expected_output"], df["actual_output"], check_names=False
    )


@pytest.mark.parametrize(
    "freqs1, freqs2, expected_output, alpha",
    [
        # Identical systems, alpha=1
        (
            [3, 2, 1],
            [3, 2, 1],
            [4 * (4 / 5) ** (1 / 2), 4 * (3 / 10) ** (1 / 2), 4 * (2 / 15) ** (1 / 2)],
            1,
        ),
        # Disjoint systems, alpha=1
        (
            [2, 1, 0, 0, 0],
            [0, 0, 3, 2, 1],
            [
                2 * (3 / 4) ** (1 / 2),
                2 * (1 / 4) ** (1 / 2),
                2 * (7 / 9) ** (1 / 2),
                2 * (5 / 18) ** (1 / 2),
                2 * (1 / 9) ** (1 / 2),
            ],
            1,
        ),
        # Disjoint systems, alpha=2
        (
            [2, 1, 0, 0, 0],
            [0, 0, 3, 2, 1],
            [
                (3 / 2) * (15 / 16) ** (1 / 3),
                (3 / 2) * (3 / 16) ** (1 / 3),
                (3 / 2) * (77 / 81) ** (1 / 3),
                (3 / 2) * (65 / 324) ** (1 / 3),
                (3 / 2) * (5 / 81) ** (1 / 3),
            ],
            2,
        ),
        # Overlapping systems, alpha=1/3
        (
            [2, 1, 0, 0],
            [0, 3, 2, 1],
            [
                4 * abs(1 - (1 / 4) ** (1 / 3)) ** (3 / 4),
                4 * abs((1 / 2) ** (1 / 3) - (1 / 4) ** (1 / 3)) ** (3 / 4)
                + 4 * abs(1 - (2 / 9) ** (1 / 3)) ** (3 / 4),
                4 * abs((1 / 2) ** (1 / 3) - (2 / 9) ** (1 / 3)) ** (3 / 4),
                4 * abs((1 / 3) ** (1 / 3) - (2 / 9) ** (1 / 3)) ** (3 / 4),
            ],
            1 / 3,
        ),
        # Overlapping systems, alpha=0
        (
            [5, 3, 2, 1, 0, 0],
            [5, 0, 4, 3, 2, 1],
            [
                log(7) + log(15 / 2),
                abs(log(2) - log(7)),
                abs(log(3) - log(7)) + abs(log(2) - log(15 / 2)),
                abs(log(4) - log(7)) + abs(log(3) - log(15 / 2)),
                abs(log(4) - log(15 / 2)),
                abs(log(5) - log(15 / 2)),
            ],
            0,
        ),
        # Overlapping systems, alpha=inf
        (
            [5, 3, 2, 1, 0, 0],
            [5, 0, 4, 3, 2, 1],
            [2.0, 1 / 2, 1 / 3 + 1 / 2, 1 / 4 + 1 / 3, 1 / 4, 1 / 5],
            float("inf"),
        ),
    ],
)
def test_rank_divergence_norm_summands(
    freqs1: list[int], freqs2: list[int], expected_output: list[float], alpha: float
) -> None:
    """ """
    df = pl.DataFrame({"freqs1": freqs1, "freqs2": freqs2, "expected_output": expected_output})
    df = df.with_columns(
        actual_output=rank_divergence_norm_summands(pl.col("freqs1"), pl.col("freqs2"), alpha=alpha)
    )

    polars.testing.assert_series_equal(
        df["expected_output"], df["actual_output"], check_names=False
    )


@pytest.mark.parametrize(
    "freqs1, freqs2, expected_output, alpha",
    [
        # Identical systems, expect no divergencen
        ([3, 2, 1], [3, 2, 1], 0, 1),
        # Disjoint, equally sized systems, expect maximum divergence
        ([2, 1, 0, 0], [0, 0, 2, 1], 1, 1),
    ],
)
def test_rank_divergence(
    freqs1: list[int], freqs2: list[int], expected_output: float, alpha: float
) -> None:
    """ """
    df = pl.DataFrame({"freqs1": freqs1, "freqs2": freqs2})
    df = df.with_columns(rank_div=rank_divergence(pl.col("freqs1"), pl.col("freqs2"), alpha=alpha))

    assert df["rank_div"].sum() == expected_output
