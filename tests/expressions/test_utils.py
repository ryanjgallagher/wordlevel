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
