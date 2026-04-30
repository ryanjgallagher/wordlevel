import pytest
import polars as pl
import polars.testing

from math import log

from wordlevel.expressions.entropy import (
    tsallis_entropy_dist_scores,
    tsallis_entropy_dual_dist_scores,
    tsallis_entropy,
    shannon_entropy,
)


@pytest.mark.parametrize(
    "input, expected_output, alpha, base, normalize",
    [
        # Shannon entropy (alpha=1), base=2
        ([2, 2, 4, 8], [3.0, 3.0, 2.0, 1.0], 1, 2, True),
        # Shannon entropy (alpha=1), base=3
        (
            [3, 1, 3, 1 / 3, 9, 1 / 3, 1, 1 / 3, 0],
            [
                log(6, 3),
                log(18, 3),
                log(6, 3),
                log(54, 3),
                log(2, 3),
                log(54, 3),
                log(18, 3),
                log(54, 3),
                float("inf"),
            ],
            1,
            3,
            True,
        ),
        # alpha=1, base=2 without normalizing frequencies
        ([2, 2, 4, 8], [-1.0, -1.0, -2.0, -3.0], 1, 2, False),
        # Tsallis entropy (alpha=2)
        ([2, 2, 4, 8], [7 / 8, 7 / 8, 3 / 4, 1 / 2], 2, None, True),
        # Tsallis entropy (alpha=3)
        ([2, 2, 4, 8], [63 / 128, 63 / 128, 15 / 32, 3 / 8], 3, None, True),
        # Tsallis entropy (alpha=1/3)
        (
            [2, 2, 4, 8, 0],
            [
                9 / 2,
                9 / 2,
                -1 * (3 / 2) * (1 - (1 / 4) ** (-2 / 3)),
                -1 * (3 / 2) * (1 - (1 / 2) ** (-2 / 3)),
                float("inf"),
            ],
            1 / 3,
            None,
            True,
        ),
    ],
)
def test_entropy_weighted_avg_scores(
    input: list[float], expected_output: list[float], alpha: float, base: float, normalize: bool
) -> None:
    """ """
    df = pl.DataFrame({"input": input, "expected_output": expected_output}, strict=False)
    df = df.with_columns(
        actual_output=tsallis_entropy_dist_scores(
            pl.col("input"), alpha=alpha, base=base, normalize=normalize
        )
    )

    polars.testing.assert_series_equal(
        df["expected_output"], df["actual_output"], check_names=False
    )


@pytest.mark.parametrize(
    "freqs1, freqs2, alpha, base",
    [([1, 2, 3, 4], [7, 5, 3, 1], 1, 2), ([1, 1, 1, 1], [1, 2, 2, 1], 3, 10)],
)
def test_tsallis_entropy_shift_scores(
    freqs1: list[int], freqs2: list[int], alpha: float, base: float
) -> None:
    """ """
    df = pl.DataFrame({"freqs1": freqs1, "freqs2": freqs2}, strict=False)
    df = df.with_columns(
        *tsallis_entropy_dual_dist_scores(
            pl.col("freqs1"),
            pl.col("freqs2"),
            base=base,
            alpha=alpha,
            alias1="scores1",
            alias2="scores2",
        ),
        entropies1=tsallis_entropy_dist_scores(pl.col("freqs1"), alpha=alpha, base=base),
        entropies2=tsallis_entropy_dist_scores(pl.col("freqs2"), alpha=alpha, base=base),
    )

    polars.testing.assert_series_equal(df["scores1"], df["entropies1"], check_names=False)
    polars.testing.assert_series_equal(df["scores2"], df["entropies2"], check_names=False)


@pytest.mark.parametrize(
    "input, expected_output, alpha, base",
    [
        # Shannon entropy (alpha=1), base=2
        ([2, 2, 4, 8, 0], [3 / 8, 3 / 8, 1 / 2, 1 / 2, 0], 1, 2),
        # Shannon entropy (alpha=1), base=3
        (
            [0, 3, 1, 3, 1 / 3, 9, 1 / 3, 1, 1 / 3, 0],
            [
                0,
                (1 / 6) * log(6, 3),
                (1 / 18) * log(18, 3),
                (1 / 6) * log(6, 3),
                (1 / 54) * log(54, 3),
                (1 / 2) * log(2, 3),
                (1 / 54) * log(54, 3),
                (1 / 18) * log(18, 3),
                (1 / 54) * log(54, 3),
                0,
            ],
            1,
            3,
        ),
        # Tsallis entropy (alpha=2)
        ([2, 2, 0, 4, 8], [7 / 64, 7 / 64, 0, 3 / 16, 1 / 4], 2, None),
        # Tsallis entropy (alpha=3)
        ([2, 2, 0, 4, 8], [63 / 1024, 63 / 1024, 0, 15 / 128, 3 / 16], 3, None),
        # Tsallis entropy (alpha=1/3)
        (
            [2, 2, 0, 4, 8],
            [
                9 / 16,
                9 / 16,
                0,
                -(3 / 2) * ((1 / 4) - (1 / 4) ** (1 / 3)),
                -(3 / 2) * ((1 / 2) - (1 / 2) ** (1 / 3)),
            ],
            1 / 3,
            None,
        ),
    ],
)
def test_tsallis_entropy(input, expected_output, alpha, base):
    """ """
    df = pl.DataFrame({"input": input, "expected_output": expected_output}, strict=False)
    df = df.with_columns(
        actual_output=tsallis_entropy(pl.col("input"), alpha=alpha, base=base),
    )

    polars.testing.assert_series_equal(
        df["expected_output"], df["actual_output"], check_names=False
    )


@pytest.mark.parametrize(
    "input, expected_output, base",
    [
        # Distribution with zero entropy
        ([1], 0, 2),
        # Uniform distribution, base=2
        ([1 / 2, 1 / 2], 1, 2),
        # Uniform distribution, base=3
        ([1 / 3, 1 / 3, 1 / 3], 1, 3),
    ],
)
def test_shannon_entropy(input, expected_output, base):
    """ """
    df = pl.DataFrame({"input": input})
    df = df.with_columns(entropy=shannon_entropy(pl.col("input"), base=base))

    assert df["entropy"].sum() == expected_output
