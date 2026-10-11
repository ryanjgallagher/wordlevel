import polars as pl
import pytest

from wordlevel.comparison.core.exclude import RankTies, freq_rank_col, rank_freqs
from wordlevel.constants import FREQ_CMP_COL, FREQ_REF_COL


class TestRankFreqs:
    @pytest.mark.parametrize(
        "ties, expected",
        [
            ("average", [1, 3, 3, 3, 5]),
            ("min", [1, 2, 2, 2, 5]),
            ("max", [1, 4, 4, 4, 5]),
            ("dense", [1, 2, 2, 2, 3]),
        ],
    )
    def test_a_tie_block_shares_one_rank_decided_by_the_tie_method(
        self, ties: RankTies, expected: list[int]
    ) -> None:
        """
        Ranks descend from the most common token, and every method holds a tie block together at a
        single shared rank. The methods disagree on which rank that is, and on whether the ranks
        after the block skip over it.
        """
        df = pl.DataFrame({FREQ_REF_COL: [10, 5, 5, 5, 1], FREQ_CMP_COL: [1, 1, 1, 1, 1]})

        df_ranked = rank_freqs(df, [ties])

        assert df_ranked[freq_rank_col(FREQ_REF_COL, ties)].to_list() == expected

    def test_each_tie_method_ranks_both_corpora(self) -> None:
        """
        A comparison can exclude by rank more than once with different tie handling, so every
        requested method contributes a column for each corpus, named after the two of them.
        """
        df = pl.DataFrame({FREQ_REF_COL: [2, 1], FREQ_CMP_COL: [1, 2]})

        df_ranked = rank_freqs(df, ["average", "dense"])

        assert df_ranked.columns == [
            FREQ_REF_COL,
            FREQ_CMP_COL,
            f"{FREQ_REF_COL}_rank_average",
            f"{FREQ_CMP_COL}_rank_average",
            f"{FREQ_REF_COL}_rank_dense",
            f"{FREQ_CMP_COL}_rank_dense",
        ]
