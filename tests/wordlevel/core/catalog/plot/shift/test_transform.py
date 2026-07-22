import pytest
import polars as pl
from polars.testing import assert_frame_equal

from wordlevel.constants import SHIFT_SCORE_TYPES
from wordlevel.core.catalog.plot.shift.transform import bar_plot_df, shift_plot_df


class TestBarPlotDf:
    def test_output_structure(self) -> None:
        """Rank filtering, output columns, and shape: 2 rows per filtered input row."""
        lf = pl.DataFrame(
            {
                "word": ["hello", "world", "foo", "bar"],
                "rank": [1, 2, 3, 51],
                "attribution": ["reference", "comparison", "reference", "comparison"],
                "score": [0.3, 0.5, -0.2, 0.1],
            }
        ).lazy()

        df_result = bar_plot_df(
            lf,
            token_col="word",
            score_col="score",
            rank_col="rank",
            attribution_col="attribution",
            contribution_alias="my_contribution",
            contribution_type_alias="my_type",
            attribution_labels=("reference", "comparison"),
            max_rank=50,
        )

        # rank=51 row excluded; 3 remaining rows * 2 attribution types = 6
        assert "bar" not in df_result["word"].to_list()
        assert len(df_result) == 6
        assert set(df_result.columns) == {"word", "rank", "my_contribution", "my_type"}

    def test_max_rank_none_keeps_every_token(self) -> None:
        """max_rank=None applies no rank filter, so every token is kept."""
        lf = pl.DataFrame(
            {
                "word": ["hello", "world", "foo", "bar"],
                "rank": [1, 2, 3, 51],
                "attribution": ["reference", "comparison", "reference", "comparison"],
                "score": [0.3, 0.5, -0.2, 0.1],
            }
        ).lazy()

        df_result = bar_plot_df(
            lf,
            token_col="word",
            score_col="score",
            rank_col="rank",
            attribution_col="attribution",
            contribution_alias="my_contribution",
            contribution_type_alias="my_type",
            attribution_labels=("reference", "comparison"),
            max_rank=None,
        )

        # No filtering: all 4 rows kept * 2 attribution types = 8
        assert set(df_result["word"].to_list()) == {"hello", "world", "foo", "bar"}
        assert len(df_result) == 8

    @pytest.mark.parametrize(
        "is_nonnegative, split_nonnegative_by_corpus, expected_ref_contribution",
        [
            # Default: no negation
            (False, True, 0.3),
            # split_nonnegative_by_corpus=False: no negation even when is_nonnegative=True
            (True, False, 0.3),
            # Both flags True: reference-attributed score is negated
            (True, True, -0.3),
        ],
    )
    def test_attribution_splitting(
        self,
        is_nonnegative: bool,
        split_nonnegative_by_corpus: bool,
        expected_ref_contribution: float,
    ) -> None:
        """Reference-attributed rows zero out the comparison column (and vice versa);
        reference scores are negated only when is_nonnegative=True and split_nonnegative_by_corpus=True."""
        lf = pl.DataFrame(
            {
                "word": ["hello", "world"],
                "rank": [1, 2],
                "attribution": ["reference", "comparison"],
                "score": [0.3, 0.5],
            }
        ).lazy()

        df_result = bar_plot_df(
            lf,
            token_col="word",
            score_col="score",
            rank_col="rank",
            attribution_col="attribution",
            contribution_alias="contribution",
            contribution_type_alias="contribution_type",
            attribution_labels=("reference", "comparison"),
            max_rank=50,
            is_nonnegative=is_nonnegative,
            split_nonnegative_by_corpus=split_nonnegative_by_corpus,
        )

        df_expected = pl.DataFrame(
            {
                "word": ["hello", "hello", "world", "world"],
                "rank": [1, 1, 2, 2],
                "contribution_type": ["reference", "comparison", "reference", "comparison"],
                "contribution": [expected_ref_contribution, 0.0, 0.0, 0.5],
            }
        )

        assert_frame_equal(df_result, df_expected, check_row_order=False, check_column_order=False)


class TestShiftPlotDf:
    def test_output_structure(self) -> None:
        """Rank filtering, output columns, shape (12 rows per filtered input row), and contribution_type values."""
        lf = pl.DataFrame(
            {
                "word": ["hello", "world", "foo"],
                "rank": [1, 2, 51],
                **{shift_type: [0.1, 0.1, 0.1] for shift_type in SHIFT_SCORE_TYPES},
            }
        ).lazy()

        df_result = shift_plot_df(
            lf,
            token_col="word",
            rank_col="rank",
            contribution_alias="my_contribution",
            contribution_type_alias="my_type",
            max_rank=50,
        )

        # rank=51 row excluded; 2 remaining rows * 12 SHIFT_SCORE_TYPES = 24
        assert "foo" not in df_result["word"].to_list()
        assert len(df_result) == 24
        assert set(df_result.columns) == {"word", "rank", "my_contribution", "my_type"}
        assert set(df_result["my_type"].to_list()) == set(SHIFT_SCORE_TYPES)

    def test_max_rank_none_keeps_every_token(self) -> None:
        """max_rank=None applies no rank filter, so every token is kept."""
        lf = pl.DataFrame(
            {
                "word": ["hello", "world", "foo"],
                "rank": [1, 2, 51],
                **{shift_type: [0.1, 0.1, 0.1] for shift_type in SHIFT_SCORE_TYPES},
            }
        ).lazy()

        df_result = shift_plot_df(
            lf,
            token_col="word",
            rank_col="rank",
            contribution_alias="my_contribution",
            contribution_type_alias="my_type",
            max_rank=None,
        )

        # No filtering: all 3 rows kept * 12 SHIFT_SCORE_TYPES = 36
        assert set(df_result["word"].to_list()) == {"hello", "world", "foo"}
        assert len(df_result) == 36
