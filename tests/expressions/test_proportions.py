import pytest
import polars as pl

from wordlevel.expressions.proportions import *


@pytest.mark.parametrize(
    "freqs1, freqs2", [([1, 2, 3, 4], [1, 2, 3, 4]), ([1, 2, 3, 4], [8, 7, 6, 5])]
)
def test_proportion_shift_scores(freqs1: list[int], freqs2: list[int]) -> None:
    """ """
    df = pl.DataFrame({"freqs1": freqs1, "freqs2": freqs2}, strict=False)
    df = df.with_columns(
        *proportion_shift_scores(
            pl.col("freqs1"), pl.col("freqs2"), alias1="scores1", alias2="scores2"
        )
    )

    assert (df["scores1"] == 1).all()
    assert (df["scores2"] == 1).all()
