import polars as pl
import pytest
from polars.testing import assert_series_equal

from wordlevel.expressions.attribution import (
    frequency_attribution,
    rank_attribution,
    score_sign_attribution,
)


@pytest.mark.parametrize(
    "normalize,expected_attributions",
    [
        (False, ["comparison", "comparison"]),
        (True, ["reference", "comparison"]),
    ],
)
def test_frequency_attribution(normalize, expected_attributions):
    """ """
    # Arrange
    df = pl.DataFrame(
        {
            "ref_col": [1.0, 3.0],  # Normalized: [0.25, 0.75]
            "cmp_col": [2.0, 8.0],  # Normalized: [0.20, 0.80]
        }
    )

    expected = pl.Series(expected_attributions, dtype=pl.Enum(["reference", "comparison"]))

    # Act
    expr = frequency_attribution(
        reference=pl.col("ref_col"), comparison=pl.col("cmp_col"), normalize=normalize
    )
    actual = df.select(expr.alias("attribution"))["attribution"]

    # Assert
    assert_series_equal(actual, expected, check_names=False)


def test_rank_attribution():
    """ """
    # Arrange
    df = pl.DataFrame(
        {
            "ref_col": [5, 4, 3, 2, 1],  # Ranks: [1, 2, 3, 4, 5]
            "cmp_col": [8, 2, 4, 6, 7],  # Ranks: [1, 5, 4, 3, 2]
        }
    )

    expected = pl.Series(
        [
            "comparison",
            "reference",
            "reference",
            "comparison",
            "comparison",
        ],
        dtype=pl.Enum(["reference", "comparison"]),
    )

    # Act
    expr = rank_attribution(reference=pl.col("ref_col"), comparison=pl.col("cmp_col"))
    actual = df.select(expr.alias("attribution"))["attribution"]

    # Assert
    assert_series_equal(actual, expected, check_names=False)


@pytest.mark.parametrize("comparison_is_positive", [True, False])
def test_score_sign_attribution(comparison_is_positive):
    """ """
    # Arrange
    df = pl.DataFrame({"score": [1, 0, -1]})

    if comparison_is_positive:
        expected = pl.Series(
            ["comparison", "comparison", "reference"],
            dtype=pl.Enum(["reference", "comparison"]),
        )
    else:
        expected = pl.Series(
            ["reference", "reference", "comparison"],
            dtype=pl.Enum(["reference", "comparison"]),
        )

    # Act
    expr = score_sign_attribution(
        score=pl.col("score"), comparison_is_positive=comparison_is_positive
    )
    actual = df.select(expr.alias("attribution"))["attribution"]

    # Assert
    assert_series_equal(actual, expected, check_names=False)
