import pytest
import polars as pl
import polars.testing
from polars import col

from wordlevel.constants import SHIFT_SCORE_TYPES
from wordlevel.expressions.shift.plan import basic_score_plan, weighted_average_score_plan


def simple_score_func(freqs1: pl.Expr, freqs2: pl.Expr) -> pl.Expr:
    return freqs2 - freqs1


def inverted_score_func(freqs1: pl.Expr, freqs2: pl.Expr) -> pl.Expr:
    """Negated relative to simple_score_func, so score sign is opposite to frequency ordering."""
    return freqs1 - freqs2


def simple_dual_score_func(
    freqs1: pl.Expr, freqs2: pl.Expr, alias1: str, alias2: str
) -> tuple[pl.Expr, pl.Expr]:
    return freqs1.alias(alias1), freqs2.alias(alias2)


def simple_clean_score_func(
    freqs1: pl.Expr,
    freqs2: pl.Expr,
    scores1: pl.Expr,
    scores2: pl.Expr,
    alias1: str,
    alias2: str,
) -> tuple[pl.Expr, pl.Expr]:
    return scores1.alias(alias1), scores2.alias(alias2)


class TestBasicScorePlan:
    def test_output_columns(self) -> None:
        """Score, rank, and attribution columns appear alongside input columns."""
        df = pl.DataFrame({"freqs_ref": [1.0, 2.0, 3.0], "freqs_cmp": [3.0, 2.0, 1.0]})

        df_result = basic_score_plan(
            df,
            freqs_ref=col("freqs_ref"),
            freqs_cmp=col("freqs_cmp"),
            score_func=simple_score_func,
            attribute_by="frequency",
            score_alias="my_score",
            rank_alias="my_rank",
            attribution_alias="my_attr",
        )

        assert "my_score" in df_result.columns
        assert "my_rank" in df_result.columns
        assert "my_attr" in df_result.columns
        assert "freqs_ref" in df_result.columns
        assert "freqs_cmp" in df_result.columns

    def test_score_values(self) -> None:
        """Score column contains the output of the score function."""
        df = pl.DataFrame({"freqs_ref": [1.0, 3.0, 8.0, 5.0], "freqs_cmp": [6.0, 2.0, 5.0, 9.0]})

        df_result = basic_score_plan(
            df,
            freqs_ref=col("freqs_ref"),
            freqs_cmp=col("freqs_cmp"),
            score_func=simple_score_func,
            attribute_by="frequency",
            score_alias="score",
        )

        # simple_score_func = freqs_cmp - freqs_ref = [5, -1, -3, 4]
        expected = pl.Series([5.0, -1.0, -3.0, 4.0])
        polars.testing.assert_series_equal(df_result["score"], expected, check_names=False)

    def test_ranking(self) -> None:
        """Rank column is the absolute-value descending rank of the score."""
        df = pl.DataFrame({"freqs_ref": [1.0, 3.0, 8.0, 5.0], "freqs_cmp": [6.0, 2.0, 5.0, 9.0]})

        df_result = basic_score_plan(
            df,
            freqs_ref=col("freqs_ref"),
            freqs_cmp=col("freqs_cmp"),
            score_func=simple_score_func,
            attribute_by="frequency",
            score_alias="score",
            rank_alias="rank",
        )

        # scores = [5, -1, -3, 4]; abs desc ranks: 5->1, 4->2, 3->3, 1->4 -> [1, 4, 3, 2]
        expected = pl.Series([1, 4, 3, 2], dtype=pl.UInt32)
        polars.testing.assert_series_equal(df_result["rank"], expected, check_names=False)

    @pytest.mark.parametrize(
        "attribute_by, score_func, expected_attribution",
        [
            (
                "frequency",
                simple_score_func,
                # cmp >= ref (normalize=False): [6>=1, 2>=3, 5>=8, 9>=5] -> [T, F, F, T]
                ["comparison", "reference", "reference", "comparison"],
            ),
            (
                "rank",
                simple_score_func,
                # ref ranks [4,3,1,2], cmp ranks [2,4,3,1]; cmp_rank <= ref_rank -> [T, F, F, T]
                ["comparison", "reference", "reference", "comparison"],
            ),
            (
                "score_sign",
                inverted_score_func,
                # inverted scores = freqs_ref - freqs_cmp = [-5, 1, 3, -4]; score >= 0 -> [F, T, T, F]
                ["reference", "comparison", "comparison", "reference"],
            ),
        ],
    )
    def test_attribution(
        self,
        attribute_by: str,
        score_func: object,
        expected_attribution: list[str],
    ) -> None:
        """The correct columns are wired to attribution for each attribute_by strategy."""
        df = pl.DataFrame({"freqs_ref": [1.0, 3.0, 8.0, 5.0], "freqs_cmp": [6.0, 2.0, 5.0, 9.0]})

        df_result = basic_score_plan(
            df,
            freqs_ref=col("freqs_ref"),
            freqs_cmp=col("freqs_cmp"),
            score_func=score_func,
            attribute_by=attribute_by,
            attribution_alias="attribution",
        )

        expected = pl.Series(expected_attribution, dtype=pl.Enum(["reference", "comparison"]))
        polars.testing.assert_series_equal(df_result["attribution"], expected, check_names=False)


class TestWeightedAverageScorePlan:
    def test_output_columns(self) -> None:
        """Output contains all required columns and none of the intermediate ones."""
        df = pl.DataFrame(
            {
                "freqs_ref": [0.4, 0.3, 0.2, 0.1],
                "freqs_cmp": [0.1, 0.2, 0.3, 0.4],
            }
        )

        df_result = weighted_average_score_plan(
            df,
            score_freqs_ref=col("freqs_ref"),
            score_freqs_cmp=col("freqs_cmp"),
            component_freqs_ref=col("freqs_ref"),
            component_freqs_cmp=col("freqs_cmp"),
            score_func=simple_dual_score_func,
            clean_score_func=simple_clean_score_func,
            attribute_by="frequency",
            score_alias="my_score",
            rank_alias="my_rank",
            attribution_alias="my_attr",
        )
        cols = set(df_result.columns)

        assert "score_ref" in cols
        assert "score_cmp" in cols
        assert "freq_diff" in cols
        assert "weighted_score_diff" in cols
        assert "ref_score_diff" in cols
        assert "my_score" in cols
        assert "my_rank" in cols
        assert "my_attr" in cols
        for shift_type in SHIFT_SCORE_TYPES:
            assert shift_type in cols, f"Missing shift score type column: {shift_type}"

        # Intermediate columns must be absent
        assert "clean_score_ref" not in cols
        assert "clean_score_cmp" not in cols
        assert "raw_score_ref" not in cols
        assert "raw_score_cmp" not in cols

    @pytest.mark.parametrize(
        "attribute_by, expected_attribution",
        [
            (
                "frequency",
                # freqs_cmp > freqs_ref -> ["comparison", "comparison"]
                # (total scores are negative with reference_score=10, so score_sign would give
                # ["reference", "reference"] -- proves frequency columns are wired, not score)
                ["comparison", "comparison"],
            ),
            (
                "score_sign",
                # total scores are negative despite freqs_cmp > freqs_ref (reference_score=10
                # dominates) -> ["reference", "reference"]
                # (frequency attribution would give ["comparison", "comparison"])
                ["reference", "reference"],
            ),
        ],
    )
    def test_attribution(self, attribute_by: str, expected_attribution: list[str]) -> None:
        """The correct columns are wired to attribution for each attribute_by strategy."""
        df = pl.DataFrame({"freqs_ref": [0.2, 0.3], "freqs_cmp": [0.8, 0.7]})

        df_result = weighted_average_score_plan(
            df,
            score_freqs_ref=col("freqs_ref"),
            score_freqs_cmp=col("freqs_cmp"),
            component_freqs_ref=col("freqs_ref"),
            component_freqs_cmp=col("freqs_cmp"),
            score_func=simple_dual_score_func,
            clean_score_func=simple_clean_score_func,
            attribute_by=attribute_by,
            reference_score=10.0,
            attribution_alias="attribution",
        )

        expected = pl.Series(expected_attribution, dtype=pl.Enum(["reference", "comparison"]))
        polars.testing.assert_series_equal(df_result["attribution"], expected, check_names=False)
