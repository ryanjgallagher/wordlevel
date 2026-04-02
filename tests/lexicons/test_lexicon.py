import pytest
import polars as pl
from polars.testing import assert_frame_equal

from wordlevel.lexicons.lexicon import join_lexicon


LABEL_COL = "word"
LEXICON_LABEL_COL = "token"
LEXICON_SCORE_COL = "sentiment"
LEXICON_SCORE_COL_IF_NOT_PROVIDED = "lexicon_score"


@pytest.fixture
def df_words() -> pl.DataFrame:
    return pl.DataFrame(
        {
            LABEL_COL: ["the", "cat", "dog", "fish"],
            "freq": [100, 50, 25, 10],
        }
    )


@pytest.fixture
def lexicon_dict() -> dict[str, float]:
    # "fish" is intentionally absent to test imputation
    return {"the": 1.0, "cat": 3.0, "dog": 7.0}


@pytest.fixture
def lexicon_df(lexicon_dict) -> pl.DataFrame:
    return pl.DataFrame(
        list(lexicon_dict.items()),
        schema=[LEXICON_LABEL_COL, LEXICON_SCORE_COL],
        orient="row",
    )


class TestJoinLexicon:
    @pytest.mark.parametrize(
        "score_col,expected_score_col", [(None, "lexicon_score"), ("my_score", "my_score")]
    )
    def test_dict_input(self, df_words, lexicon_dict, score_col, expected_score_col):
        result = join_lexicon(
            df_words, lexicon_dict, df_label_col=LABEL_COL, lexicon_score_col=score_col
        )

        expected = pl.DataFrame(
            {
                LABEL_COL: ["the", "cat", "dog", "fish"],
                expected_score_col: [1.0, 3.0, 7.0, 0.0],
            }
        )

        assert_frame_equal(
            result.select(LABEL_COL, expected_score_col), expected, check_row_order=False
        )
        assert "freq" in result.columns

    def test_dataframe_input(self, df_words, lexicon_df):
        result = join_lexicon(
            df_words,
            lexicon_df,
            df_label_col=LABEL_COL,
            lexicon_label_col=LEXICON_LABEL_COL,
            lexicon_score_col=LEXICON_SCORE_COL,
        )

        expected = pl.DataFrame(
            {
                LABEL_COL: ["the", "cat", "dog", "fish"],
                LEXICON_SCORE_COL: [1.0, 3.0, 7.0, 0.0],
            }
        )

        assert_frame_equal(
            result.select(LABEL_COL, LEXICON_SCORE_COL), expected, check_row_order=False
        )
        assert "freq" in result.columns


class TestImputation:
    @pytest.mark.parametrize(
        "impute, expected_fish_score",
        [
            (0.0, 0.0),
            (5.0, 5.0),
            ("mean", (1.0 + 3.0 + 7.0) / 3),
            (pl.lit(2.5), 2.5),
        ],
    )
    def test_missing_words_are_imputed(self, df_words, lexicon_dict, impute, expected_fish_score):
        result = join_lexicon(df_words, lexicon_dict, df_label_col=LABEL_COL, impute=impute)

        expected = pl.DataFrame(
            {
                LABEL_COL: ["the", "cat", "dog", "fish"],
                LEXICON_SCORE_COL_IF_NOT_PROVIDED: [1.0, 3.0, 7.0, expected_fish_score],
            }
        )
        assert_frame_equal(
            result.select(LABEL_COL, LEXICON_SCORE_COL_IF_NOT_PROVIDED),
            expected,
            check_row_order=False,
        )


class TestValidation:
    def test_dataframe_lexicon_requires_lexicon_label_col(self, df_words, lexicon_df):
        with pytest.raises(ValueError, match="lexicon_label_col"):
            join_lexicon(df_words, lexicon_df, df_label_col=LABEL_COL)

    def test_dataframe_lexicon_requires_lexicon_score_col(self, df_words, lexicon_df):
        with pytest.raises(ValueError, match="lexicon_score_col"):
            join_lexicon(
                df_words, lexicon_df, df_label_col=LABEL_COL, lexicon_label_col=LEXICON_LABEL_COL
            )


class TestReturnType:
    @pytest.mark.parametrize(
        "as_lazy, expected_type",
        [
            (False, pl.DataFrame),
            (True, pl.LazyFrame),
        ],
    )
    def test_return_type_matches_input(self, df_words, lexicon_dict, as_lazy, expected_type):
        df_input = df_words.lazy() if as_lazy else df_words
        result = join_lexicon(df_input, lexicon_dict, df_label_col=LABEL_COL)
        assert isinstance(result, expected_type)
