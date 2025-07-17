import pytest
import polars as pl 
import polars.testing 

from wordlevel.expressions.utils import *


@pytest.mark.parametrize('input, expected_output, normalize', [
    # Test normalization works as expected
    (
        [1, 2, 3, 4],
        [1 / 10, 2 / 10, 3 / 10, 4 / 10],
        True
    ),
    # Test normalization returns an already normalized distribution as expected
    (
        [1 / 6, 2 / 6, 3 / 6],
        [1 / 6, 2 / 6, 3 / 6],
        True
    ),
    # Test normalization does not return normalization if not asked for
    (
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        False
    )
])
def test_normalize_freqs(
    input: list[float], 
    expected_output: list[float],
    normalize: bool
) -> None:
    """
    """
    df = pl.DataFrame({'input': input, 'expected_output': expected_output}, strict=False)
    df = df.with_columns(actual_output=normalize_freqs(pl.col('input'), normalize=normalize))
    
    polars.testing.assert_series_equal(
        df['expected_output'], 
        df['actual_output'],
        check_names=False
    )

@pytest.mark.parametrize('probs1, probs2, weight1, weight2, expected_output', [
    # Equally mixed distributions
    (
        [1 / 8, 3 / 8, 2 / 8, 2 / 8],
        [2 / 10, 3 / 10, 1 / 10, 4 / 10],
        1 / 2,
        1 / 2,
        [13 / 80, 27 / 80, 14 / 80, 26/ 80]
    ),
    # Unequally mixed distributions
    (
        [1 / 8, 3 / 8, 2 / 8, 2 / 8],
        [2 / 10, 3 / 10, 1 / 10, 4 / 10],
        1 / 4,
        3 / 4,
        [29 / 160, 51 / 160, 22 / 160, 58 / 160]
    ),
    # Unequally mixed distributions, weights that aren't normalized to 1
    (
        [1 / 8, 3 / 8, 2 / 8, 2 / 8],
        [2 / 10, 3 / 10, 1 / 10, 4 / 10],
        1,
        2,
        [7 / 40, 13 / 40, 6 / 40, 14 / 40]
    )
])
def test_mixed_distribution(
    probs1: list[float],
    probs2: list[float],
    weight1: float,
    weight2: float,
    expected_output: list[float]
) -> None:
    """
    """
    df = pl.DataFrame(
        {'probs1': probs1, 'probs2': probs2, 'expected_output': expected_output}, 
        strict=False
    )
    df = df.with_columns(
        actual_output=mixed_distribution(
            pl.col('probs1'), 
            pl.col('probs2'), 
            weight1=weight1,
            weight2=weight2
        )
    )
    
    polars.testing.assert_series_equal(
        df['expected_output'], 
        df['actual_output'],
        check_names=False
    )