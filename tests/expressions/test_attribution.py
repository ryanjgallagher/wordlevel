import polars as pl
import pytest
from polars.testing import assert_series_equal

from wordlevel.expressions.attribution import (
    frequency_attribution,
    rank_attribution,
    REFERENCE_ATTRIBUTION,
    COMPARISON_ATTRIBUTION,
)


@pytest.mark.parametrize(
    "normalize,expected_attributions",
    [
        (False, [COMPARISON_ATTRIBUTION, COMPARISON_ATTRIBUTION]),
        (True, [REFERENCE_ATTRIBUTION, COMPARISON_ATTRIBUTION]),
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

    expected = pl.Series(expected_attributions)

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
            COMPARISON_ATTRIBUTION,
            REFERENCE_ATTRIBUTION,
            REFERENCE_ATTRIBUTION,
            COMPARISON_ATTRIBUTION,
            COMPARISON_ATTRIBUTION,
        ]
    )

    # Act
    expr = rank_attribution(reference=pl.col("ref_col"), comparison=pl.col("cmp_col"))
    actual = df.select(expr.alias("attribution"))["attribution"]

    # Assert
    assert_series_equal(actual, expected, check_names=False)
