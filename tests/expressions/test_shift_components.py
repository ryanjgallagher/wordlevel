import pytest
import polars as pl
import polars.testing

from wordlevel.expressions.shift_components import (
    frequency_diff,
    weighted_score_diff,
    reference_score_diff,
    get_shift_components,
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


@pytest.mark.parametrize("include_total_difference", [True, False])
def test_get_shift_components(include_total_difference: bool) -> None:
    """ """
    reference_score = 8.0

    # fmt: off
    df_input = pl.DataFrame({
        "f1": [10, 10],  # Normalizes to [0.5, 0.5]
        "f2": [20, 60],  # Normalizes to [0.25, 0.75]
        "s1": [5, 5],
        "s2": [10, 15],
    })

    df_actual = get_shift_components(
        df_input,
        freqs1=pl.col("f1"),
        freqs2=pl.col("f2"),
        scores1=pl.col("s1"),
        scores2=pl.col("s2"),
        reference_score=reference_score,
        frequency_diff_alias='diff_of_freqs',
        weighted_score_diff_alias='diff_of_weighted_scores',
        reference_score_diff_alias='diff_with_reference_score',
        total_difference_alias='overall_difference',
        include_total_difference=include_total_difference
    )

    df_expected = pl.DataFrame({
        "f1": [10, 10],
        "f2": [20, 60],
        "s1": [5, 5],
        "s2": [10, 15],
        "diff_of_freqs": [-0.25, 0.25],            # [0.25 - 0.5, 0.75 - 0.5]
        "diff_of_weighted_scores": [1.875, 6.25],  # [0.375 * 5, 0.625 * 10]
        "diff_with_reference_score": [-0.5, 2.0],  # [7.5 - 8.0, 10.0 - 8.0]
        "overall_difference": [2.0, 6.75],         # [(-0.25 * -0.5) + 1.875, (0.25 * 2.0) + 6.25]
    })
    if not include_total_difference:
        df_expected = df_expected.drop('overall_difference')
    # fmt: on

    polars.testing.assert_frame_equal(df_actual, df_expected)


@pytest.mark.parametrize("normalize", [True, False])
def test_get_shift_components_normalization(normalize: bool) -> None:
    """ """
    reference_score = 8.0

    df_input = pl.DataFrame(
        {
            "f1": [10.0, 10.0],
            "f2": [20.0, 60.0],
            "s1": [5, 5],
            "s2": [10, 15],
        }
    )

    df_actual = get_shift_components(
        df_input,
        freqs1=pl.col("f1"),
        freqs2=pl.col("f2"),
        scores1=pl.col("s1"),
        scores2=pl.col("s2"),
        frequency_diff_alias="freq_diff",
        reference_score=reference_score,
        normalize=normalize,
    )

    if normalize:
        expected_freq_diff = [-0.25, 0.25]
    else:
        expected_freq_diff = [10.0, 50.0]

    # Only check the frequency difference to verify the flag worked
    polars.testing.assert_series_equal(
        df_actual["freq_diff"], pl.Series(expected_freq_diff), check_names=False
    )
