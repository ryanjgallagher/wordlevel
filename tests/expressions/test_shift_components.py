import pytest
import polars as pl
import polars.testing

from wordlevel.expressions.shift_components import (
    frequency_diff,
    weighted_score_diff,
    reference_score_diff,
)


@pytest.mark.parametrize(
    "freqs1, freqs2, expected",
    [
        ([1, 2, 3, 4, 5], [4, 3, 2, 1, 5], [3 / 15, 1 / 15, -1 / 15, -3 / 15, 0.0]),
        ([1, 1, 2, 2], [1, 1, 1, 3], [0.0, 0.0, -1 / 6, 1 / 6]),
    ],
)
def test_frequency_diff(freqs1: list[float], freqs2: list[float], expected: list[float]) -> None:
    """ """
    df = pl.DataFrame({"freqs1": freqs1, "freqs2": freqs2, "expected": expected}, strict=False)
    df = df.with_columns(actual=frequency_diff(pl.col("freqs1"), pl.col("freqs2")))

    polars.testing.assert_series_equal(df["actual"], df["expected"], check_names=False)


@pytest.mark.parametrize(
    "freqs1, freqs2, scores1, scores2, expected",
    [
        (
            [1 / 10, 2 / 10, 3 / 10, 4 / 10],
            [2 / 10, 4 / 10, 0.0, 4 / 10],
            [100, 500, 1000, 400],
            [800, 900, 600, 300],
            [105.0, 120.0, -60.0, -40.0],
        ),
        (
            [1 / 6, 1 / 6, 2 / 6, 2 / 6],
            [1 / 6, 1 / 6, 1 / 6, 3 / 6],
            [18, 42, 42, 60],
            [12, 30, 42, 60],
            [-1.0, -2.0, 0.0, 0.0],
        ),
    ],
)
def test_weighted_score_diff(
    freqs1: list[float],
    freqs2: list[float],
    scores1: list[float],
    scores2: list[float],
    expected: list[float],
) -> None:
    """ """
    df = pl.DataFrame(
        {
            "freqs1": freqs1,
            "freqs2": freqs2,
            "scores1": scores1,
            "scores2": scores2,
            "expected": expected,
        },
        strict=False,
    )
    df = df.with_columns(
        actual=weighted_score_diff(
            pl.col("freqs1"), pl.col("freqs2"), pl.col("scores1"), pl.col("scores2")
        )
    )

    polars.testing.assert_series_equal(
        df["actual"],
        df["expected"],
        check_names=False,
    )


@pytest.mark.parametrize(
    "scores1, scores2, ref_score, expected",
    [
        ([1, 2, 17, 1], [3, 8, 15, 1], 5, [-3.0, 0.0, 11.0, -4.0]),
        ([6, 7, 8, 9], [4, 9, 6, 11], 0, [5.0, 8.0, 7.0, 10.0]),
    ],
)
def test_reference_score_diff(
    scores1: list[float], scores2: list[float], ref_score: float, expected: list[float]
) -> None:
    """ """
    df = pl.DataFrame({"scores1": scores1, "scores2": scores2, "expected": expected}, strict=False)
    df = df.with_columns(
        actual=reference_score_diff(pl.col("scores1"), pl.col("scores2"), ref_score)
    )

    polars.testing.assert_series_equal(df["actual"], df["expected"], check_names=False)
