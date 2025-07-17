import pytest
import polars as pl 
import polars.testing 

from math import log10

from wordlevel.expressions.shift_scores import * 


@pytest.mark.parametrize('freqs1, freqs2', [
    ([1, 2, 3, 4], [1, 2, 3, 4]),
    ([1, 2, 3, 4], [8, 7, 6, 5])
])
def test_proportion_shift_scores(freqs1: list[int], freqs2: list[int]) -> None:
    """
    """
    df = pl.DataFrame({'freqs1': freqs1, 'freqs2': freqs2}, strict=False)
    df = df.with_columns(
        *proportion_shift_scores(
            pl.col('freqs1'), 
            pl.col('freqs2'),
            alias1='scores1',
            alias2='scores2'
        )
    )
    
    assert (df['scores1'] == 1).all()
    assert (df['scores2'] == 1).all()

@pytest.mark.parametrize('freqs1, freqs2, alpha, base', [
    ([1, 2, 3, 4], [7, 5, 3, 1], 1, 2),
    ([1, 1, 1, 1], [1, 2, 2, 1], 3, 10)
])
def test_tsallis_entropy_shift_scores(
    freqs1: list[int], 
    freqs2: list[int], 
    alpha: float, 
    base: float
) -> None:
    """
    """
    df = pl.DataFrame({'freqs1': freqs1, 'freqs2': freqs2}, strict=False)
    df = df.with_columns(
        *tsallis_entropy_shift_scores(
            pl.col('freqs1'), 
            pl.col('freqs2'),
            base=base,
            alpha=alpha,
            alias1='scores1',
            alias2='scores2'
        ),
        entropies1=tsallis_entropy_weighted_avg_scores(pl.col('freqs1'), alpha=alpha, base=base),
        entropies2=tsallis_entropy_weighted_avg_scores(pl.col('freqs2'), alpha=alpha, base=base),
    )

    polars.testing.assert_series_equal(df['scores1'], df['entropies1'], check_names=False)
    polars.testing.assert_series_equal(df['scores2'], df['entropies2'], check_names=False)

@pytest.mark.parametrize('freqs1, freqs2, base', [
    ([1, 2, 3, 4], [7, 5, 3, 1], 2),
    ([1, 1, 1, 1], [1, 2, 2, 1], 10)
])
def test_kld_shift_scores(freqs1: list[int], freqs2: list[int], base: float) -> None:
    """
    """
    df = pl.DataFrame({'freqs1': freqs1, 'freqs2': freqs2}, strict=False)
    df = df.with_columns(
        *kullback_leibler_divergence_shift_scores(
            pl.col('freqs1'), 
            pl.col('freqs2'),
            base=base,
            alias1='scores1',
            alias2='scores2'
        ),
        entropies1=tsallis_entropy_weighted_avg_scores(pl.col('freqs1'), base=base),
        entropies2=tsallis_entropy_weighted_avg_scores(pl.col('freqs2'), base=base),
    )

    polars.testing.assert_series_equal(df['scores1'], df['entropies1'], check_names=False)
    polars.testing.assert_series_equal(df['scores2'], df['entropies2'], check_names=False)

jsd_params = 'freqs1, freqs2, alpha, base, weight1, weight2, expected_scores1, expected_scores2'
@pytest.mark.parametrize(jsd_params, [
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
            (1 / 2) * log10(13 / 10)
        ],
        [
            (1 / 2) * log10(16 / 13),
            (1 / 2) * log10(8 / 9),
            (1 / 2) * log10(4 / 7),
            (1 / 2) * log10(16 / 13)
        ]
    ),
    # Unequally mixed distributions, alpha=3, 
    (
        [1 / 8, 3 / 8, 2 / 8, 2 / 8],
        [2 / 10, 3 / 10, 1 / 10, 4 / 10],
        3,
        10,
        1 / 4,
        3 / 4,
        [
            441 / 204800,
            -999 / 204800,
            -279 / 51200,
            441 / 51200
        ],
        [
            549 / 204800,
            -891 / 204800,
            -171 / 51200,
            549 / 51200
        ]
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
            (1 / 3) * ( (7 / 40) ** (-2 / 3) - (1 / 8) ** (-2 / 3) ) / (-2 / 3),
            (1 / 3) * ( (13 / 40) ** (-2 / 3) - (3 / 8) ** (-2 / 3) ) / (-2 / 3),
            (1 / 3) * ( (6 / 40) ** (-2 / 3) - (2 / 8) ** (-2 / 3) ) / (-2 / 3),
            (1 / 3) * ( (14 / 40) ** (-2 / 3) - (2 / 8) ** (-2 / 3) ) / (-2 / 3)
        ],
        [
            (2 / 3) * ( (2 / 10) ** (-2 / 3) - (7 / 40) ** (-2 / 3) ) / (-2 / 3),
            (2 / 3) * ( (3 / 10) ** (-2 / 3) - (13 / 40) ** (-2 / 3) ) / (-2 / 3),
            (2 / 3) * ( (1 / 10) ** (-2 / 3) - (6 / 40) ** (-2 / 3) ) / (-2 / 3),
            (2 / 3) * ( (4 / 10) ** (-2 / 3) - (14 / 40) ** (-2 / 3) ) / (-2 / 3),
        ]
    )
])
def test_jsd_shift_scores(
    freqs1: list[int],
    freqs2: list[int],
    alpha: float,
    base: float,
    weight1: float,
    weight2: float,
    expected_scores1: list[int],
    expected_scores2: list[int]
) -> None:
    """
    """
    df = pl.DataFrame(
        {
            'freqs1': freqs1, 
            'freqs2': freqs2,
            'expected_scores1': expected_scores1,
            'expected_scores2': expected_scores2
        }, 
        strict=False
    )
    df = df.with_columns(
        *jensen_shannon_divergence_shift_scores(
            pl.col('freqs1'), 
            pl.col('freqs2'),
            base=base,
            alpha=alpha,
            weight1=weight1,
            weight2=weight2,
            alias1='scores1',
            alias2='scores2'
        ),
    )

    polars.testing.assert_series_equal(df['scores1'], df['expected_scores1'], check_names=False)
    polars.testing.assert_series_equal(df['scores2'], df['expected_scores2'], check_names=False)