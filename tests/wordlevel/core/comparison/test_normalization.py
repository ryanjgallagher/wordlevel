import pytest
import polars as pl
import polars.testing

from wordlevel.core.comparison.normalization import (
    normalization_constant_expr,
    normalize_scores,
)


SCORE_COL = "comparison_score"
CONSTANT_COL = "normalization_constant"


class TestNormalizationConstantExpr:
    @pytest.mark.parametrize(
        "by, scores, expected",
        [
            ("sum_abs", [2.0, -3.0, 1.0], 6.0),
            ("abs_sum", [2.0, -3.0, 1.0], 0.0),
            ("sum_abs", [2.0, -1.0, 1.0], 4.0),
            ("abs_sum", [2.0, -1.0, 1.0], 2.0),
        ],
    )
    def test_constant(self, by: str, scores: list[float], expected: float) -> None:
        """The divisor is the sum of absolutes for `sum_abs` and the absolute sum for `abs_sum`."""
        df = pl.DataFrame({SCORE_COL: scores})
        constant = df.select(normalization_constant_expr(by, SCORE_COL)).item()  # type: ignore[arg-type]

        assert constant == expected


class TestNormalizeScores:
    def test_rescales_score_and_adds_constant(self) -> None:
        """`score_col` is divided by the constant, which is also written as its own column."""
        df = pl.DataFrame({SCORE_COL: [2.0, -1.0, 1.0]})

        df_norm = normalize_scores(df, "sum_abs", SCORE_COL, [], CONSTANT_COL)
        expected_scores = pl.Series(SCORE_COL, [2 / 4, -1 / 4, 1 / 4])

        polars.testing.assert_series_equal(df_norm[SCORE_COL], expected_scores)
        assert df_norm[CONSTANT_COL].to_list() == [4.0, 4.0, 4.0]

    def test_decomposition_preserved(self) -> None:
        """Component columns are scaled by the same constant, so they still sum to the total."""
        df = pl.DataFrame(
            {
                SCORE_COL: [3.0, -2.0],
                "more_of_positive": [2.0, -3.0],
                "higher_score": [1.0, 1.0],
            }
        )

        df_norm = normalize_scores(
            df, "sum_abs", SCORE_COL, ["more_of_positive", "higher_score"], CONSTANT_COL
        )

        summed_components = df_norm.select(
            pl.sum_horizontal("more_of_positive", "higher_score")
        ).to_series()
        polars.testing.assert_series_equal(summed_components, df_norm[SCORE_COL], check_names=False)
