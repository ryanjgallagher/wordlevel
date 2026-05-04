import pytest
import polars as pl
import polars.testing
from polars import col
from polars.testing import assert_series_equal, assert_series_not_equal

from wordlevel.constants import SHIFT_SCORE_TYPES
from wordlevel.expressions.lexicon import DEFAULT_LEXICON_SCORE_COL
from wordlevel.expressions.shift.plan import (
    basic_score_plan,
    weighted_average_score_plan,
    lexicon_join_plan,
)
from wordlevel.lexicons.lexicon import Lexicon


_LABEL_COL = "word"
_LEXICON_LABEL_COL = "token"
_LEXICON_SCORE_COL = "sentiment"


@pytest.fixture
def df_words() -> pl.DataFrame:
    return pl.DataFrame(
        {
            _LABEL_COL: ["the", "cat", "dog", "fish"],
            "freq": [100, 50, 25, 10],
        }
    )


@pytest.fixture
def lexicon_dict() -> dict[str, float]:
    return {"the": 1.0, "cat": 3.0, "dog": 7.0}


@pytest.fixture
def lexicon_dict_alt() -> dict[str, float]:
    return {"the": 2.0, "cat": 4.0, "dog": 5.0, "fish": 9.0}


@pytest.fixture
def lexicon_df(lexicon_dict) -> pl.DataFrame:
    return pl.DataFrame(
        list(lexicon_dict.items()),
        schema=[_LEXICON_LABEL_COL, _LEXICON_SCORE_COL],
        orient="row",
    )


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
        df = pl.DataFrame(
            {"word": ["a", "b", "c"], "freqs_ref": [1.0, 2.0, 3.0], "freqs_cmp": [3.0, 2.0, 1.0]}
        )

        df_result = basic_score_plan(
            df,
            freqs_ref=col("freqs_ref"),
            freqs_cmp=col("freqs_cmp"),
            label=col("word"),
            score_func=simple_score_func,
            attribute_by="frequency",
            score_alias="my_score",
            rank_alias="my_rank",
            attribution_alias="my_attr",
        )

        assert "word" in df_result.columns
        assert "my_score" in df_result.columns
        assert "my_rank" in df_result.columns
        assert "my_attr" in df_result.columns
        assert "freqs_ref" in df_result.columns
        assert "freqs_cmp" in df_result.columns

    def test_score_values(self) -> None:
        """Score column contains the output of the score function."""
        df = pl.DataFrame(
            {
                "token": ["a", "b", "c", "d"],
                "freqs_ref": [1.0, 3.0, 8.0, 5.0],
                "freqs_cmp": [6.0, 2.0, 5.0, 9.0],
            }
        )

        df_result = basic_score_plan(
            df,
            freqs_ref=col("freqs_ref"),
            freqs_cmp=col("freqs_cmp"),
            label=col("token"),
            score_func=simple_score_func,
            attribute_by="frequency",
            score_alias="score",
        )

        # simple_score_func = freqs_cmp - freqs_ref = [5, -1, -3, 4]
        expected = pl.Series([5.0, -1.0, -3.0, 4.0])
        assert_series_equal(df_result["score"], expected, check_names=False)

    def test_ranking(self) -> None:
        """Rank column is the absolute-value descending rank of the score."""
        df = pl.DataFrame(
            {
                "word": ["a", "b", "c", "d"],
                "freqs_ref": [1.0, 3.0, 8.0, 5.0],
                "freqs_cmp": [6.0, 2.0, 5.0, 9.0],
            }
        )

        df_result = basic_score_plan(
            df,
            freqs_ref=col("freqs_ref"),
            freqs_cmp=col("freqs_cmp"),
            label=col("word"),
            score_func=simple_score_func,
            attribute_by="frequency",
            score_alias="score",
            rank_alias="rank",
        )

        # scores = [5, -1, -3, 4]; abs desc ranks: 5->1, 4->2, 3->3, 1->4 -> [1, 4, 3, 2]
        expected = pl.Series([1, 4, 3, 2], dtype=pl.UInt32)
        assert_series_equal(df_result["rank"], expected, check_names=False)

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
        df = pl.DataFrame(
            {
                "word": ["a", "b", "c", "d"],
                "freqs_ref": [1.0, 3.0, 8.0, 5.0],
                "freqs_cmp": [6.0, 2.0, 5.0, 9.0],
            }
        )

        df_result = basic_score_plan(
            df,
            freqs_ref=col("freqs_ref"),
            freqs_cmp=col("freqs_cmp"),
            label=col("word"),
            score_func=score_func,
            attribute_by=attribute_by,
            attribution_alias="attribution",
        )

        expected = pl.Series(expected_attribution, dtype=pl.Enum(["reference", "comparison"]))
        assert_series_equal(df_result["attribution"], expected, check_names=False)


class TestWeightedAverageScorePlan:
    def test_output_columns(self) -> None:
        """Output contains all required columns and none of the intermediate ones."""
        df = pl.DataFrame(
            {
                "token": ["a", "b", "c", "d"],
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
            label=col("token"),
            score_func=simple_dual_score_func,
            clean_score_func=simple_clean_score_func,
            attribute_by="frequency",
            score_alias="my_score",
            rank_alias="my_rank",
            attribution_alias="my_attr",
        )
        cols = set(df_result.columns)

        assert "token" in cols
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
        df = pl.DataFrame({"word": ["a", "b"], "freqs_ref": [0.2, 0.3], "freqs_cmp": [0.8, 0.7]})

        df_result = weighted_average_score_plan(
            df,
            score_freqs_ref=col("freqs_ref"),
            score_freqs_cmp=col("freqs_cmp"),
            component_freqs_ref=col("freqs_ref"),
            component_freqs_cmp=col("freqs_cmp"),
            label=col("word"),
            score_func=simple_dual_score_func,
            clean_score_func=simple_clean_score_func,
            attribute_by=attribute_by,
            reference_score=10.0,
            attribution_alias="attribution",
        )

        expected = pl.Series(expected_attribution, dtype=pl.Enum(["reference", "comparison"]))
        assert_series_equal(df_result["attribution"], expected, check_names=False)


class TestLexiconJoinPlan:
    @pytest.mark.parametrize(
        "lexicon_score_col, score_alias_ref, score_alias_cmp, expected_alias_ref, expected_alias_cmp",
        [
            # All None: aliases derive from DEFAULT_LEXICON_SCORE_COL
            (
                None,
                None,
                None,
                f"{DEFAULT_LEXICON_SCORE_COL}_ref",
                f"{DEFAULT_LEXICON_SCORE_COL}_cmp",
            ),
            # Custom aliases override everything
            (None, "my_ref", "my_cmp", "my_ref", "my_cmp"),
            # Custom lexicon_score_col with no aliases: aliases derive from it
            (
                _LEXICON_SCORE_COL,
                None,
                None,
                f"{_LEXICON_SCORE_COL}_ref",
                f"{_LEXICON_SCORE_COL}_cmp",
            ),
        ],
    )
    def test_dict_lexicon_output(
        self,
        df_words,
        lexicon_dict,
        lexicon_score_col,
        score_alias_ref,
        score_alias_cmp,
        expected_alias_ref,
        expected_alias_cmp,
    ) -> None:
        """ """
        result = lexicon_join_plan(
            df_words,
            df_label_col=_LABEL_COL,
            lexicon=lexicon_dict,
            lexicon_score_col=lexicon_score_col,
            score_alias_ref=score_alias_ref,
            score_alias_cmp=score_alias_cmp,
        )

        assert expected_alias_ref in result.columns
        assert expected_alias_cmp in result.columns
        assert _LABEL_COL in result.columns
        assert "freq" in result.columns

    @pytest.mark.parametrize(
        "score_alias_ref, score_alias_cmp, expected_alias_ref, expected_alias_cmp",
        [
            # All None: aliases derive from lexicon.score_col
            (None, None, "score_ref", "score_cmp"),
            # Custom aliases override everything
            ("my_ref", "my_cmp", "my_ref", "my_cmp"),
        ],
    )
    def test_lexicon_input(
        self,
        df_words,
        score_alias_ref,
        score_alias_cmp,
        expected_alias_ref,
        expected_alias_cmp,
    ) -> None:
        lexicon = Lexicon(
            lf=pl.LazyFrame({"token": ["the", "cat", "dog"], "score": [1.0, 3.0, 7.0]}),
            center=5.0,
        )
        result = lexicon_join_plan(
            df_words,
            df_label_col=_LABEL_COL,
            lexicon=lexicon,
            score_alias_ref=score_alias_ref,
            score_alias_cmp=score_alias_cmp,
        )

        assert expected_alias_ref in result.columns
        assert expected_alias_cmp in result.columns
        assert _LABEL_COL in result.columns
        assert "freq" in result.columns

    @pytest.mark.parametrize(
        "score_alias_ref, score_alias_cmp, expected_alias_ref, expected_alias_cmp",
        [
            # All None: aliases derive from DEFAULT_LEXICON_SCORE_COL
            (None, None, f"{_LEXICON_SCORE_COL}_ref", f"{_LEXICON_SCORE_COL}_cmp"),
            # Custom aliases override everything
            ("my_ref", "my_cmp", "my_ref", "my_cmp"),
        ],
    )
    def test_dataframe_lexicon_input(
        self,
        df_words,
        lexicon_df,
        score_alias_ref,
        score_alias_cmp,
        expected_alias_ref,
        expected_alias_cmp,
    ) -> None:
        """ """
        result = lexicon_join_plan(
            df_words,
            df_label_col=_LABEL_COL,
            lexicon=lexicon_df,
            lexicon_label_col=_LEXICON_LABEL_COL,
            lexicon_score_col=_LEXICON_SCORE_COL,
            score_alias_ref=score_alias_ref,
            score_alias_cmp=score_alias_cmp,
        )

        assert expected_alias_ref in result.columns
        assert expected_alias_cmp in result.columns
        assert _LABEL_COL in result.columns
        assert "freq" in result.columns

    @pytest.mark.parametrize(
        "lexicon_keyword",
        [("lexicon"), ("lexicon_ref"), ("lexicon_cmp")],
    )
    def test_single_lexicon_scores_are_identical(
        self, df_words, lexicon_dict, lexicon_keyword
    ) -> None:
        """When a single lexicon is used for both sides, ref and cmp scores are identical."""
        result = lexicon_join_plan(
            df_words,
            df_label_col=_LABEL_COL,
            score_alias_ref="score_ref",
            score_alias_cmp="score_cmp",
            **{lexicon_keyword: lexicon_dict},
        )

        assert_series_equal(result["score_ref"], result["score_cmp"], check_names=False)

    def test_separate_lexicons_give_different_scores(
        self, df_words, lexicon_dict, lexicon_dict_alt
    ) -> None:
        """When `lexicon_ref` and `lexicon_cmp` differ, scores in each column differ."""
        result = lexicon_join_plan(
            df_words,
            df_label_col=_LABEL_COL,
            lexicon_ref=lexicon_dict,
            lexicon_cmp=lexicon_dict_alt,
            score_alias_ref="score_ref",
            score_alias_cmp="score_cmp",
        )

        assert_series_not_equal(result["score_ref"], result["score_cmp"], check_names=False)

    class TestValidation:
        def test_no_lexicon_raises(self, df_words) -> None:
            with pytest.raises(ValueError, match="lexicon"):
                lexicon_join_plan(df_words, df_label_col=_LABEL_COL)

        @pytest.mark.parametrize(
            "lexicon_keyword",
            [("lexicon_ref"), ("lexicon_cmp")],
        )
        def test_lexicon_and_lexicon_ref_raises(
            self, df_words, lexicon_dict, lexicon_keyword
        ) -> None:
            with pytest.raises(ValueError, match="ambiguous"):
                lexicon_join_plan(
                    df_words,
                    df_label_col=_LABEL_COL,
                    lexicon=lexicon_dict,
                    **{lexicon_keyword: lexicon_dict},
                )

        def test_lexicon_and_lexicon_cmp_raises(self, df_words, lexicon_dict) -> None:
            with pytest.raises(ValueError, match="ambiguous"):
                lexicon_join_plan(
                    df_words,
                    df_label_col=_LABEL_COL,
                    lexicon=lexicon_dict,
                    lexicon_cmp=lexicon_dict,
                )
