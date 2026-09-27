import polars as pl
import pytest
from polars.testing import assert_frame_equal

from wordlevel.core.comparison.score.lexicon import (
    DEFAULT_LEXICON_SCORE_COL,
    get_imputation_expression,
    join_lexicon,
)
from wordlevel.core.lexicons.load import Lexicon


TOKEN_COL = "word"
LEXICON_TOKEN_COL = "token"
LEXICON_SCORE_COL = "sentiment"


@pytest.fixture
def df_words() -> pl.DataFrame:
    return pl.DataFrame(
        {
            TOKEN_COL: ["the", "cat", "dog", "fish"],
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
        schema=[LEXICON_TOKEN_COL, LEXICON_SCORE_COL],
        orient="row",
    )


class TestJoinLexicon:
    @pytest.mark.parametrize(
        "score_col,expected_score_col",
        [(None, DEFAULT_LEXICON_SCORE_COL), ("my_score", "my_score")],
    )
    def test_dict_input(self, df_words, lexicon_dict, score_col, expected_score_col):
        result = join_lexicon(
            df_words, lexicon_dict, df_token_col=TOKEN_COL, lexicon_score_col=score_col
        )

        expected = pl.DataFrame(
            {
                TOKEN_COL: ["the", "cat", "dog", "fish"],
                expected_score_col: [1.0, 3.0, 7.0, None],
            }
        )

        assert_frame_equal(
            result.select(TOKEN_COL, expected_score_col), expected, check_row_order=False
        )
        assert "freq" in result.columns

    def test_lexicon_input(self, df_words):
        lexicon = Lexicon(
            lf=pl.LazyFrame({"token": ["the", "cat", "dog"], "score": [1.0, 3.0, 7.0]}),
            center=5.0,
        )
        result = join_lexicon(df_words, lexicon, df_token_col=TOKEN_COL)

        expected = pl.DataFrame(
            {
                TOKEN_COL: ["the", "cat", "dog", "fish"],
                "score": [1.0, 3.0, 7.0, None],
            }
        )

        assert_frame_equal(result.select(TOKEN_COL, "score"), expected, check_row_order=False)
        assert "freq" in result.columns

    def test_dataframe_input(self, df_words, lexicon_df):
        result = join_lexicon(
            df_words,
            lexicon_df,
            df_token_col=TOKEN_COL,
            lexicon_token_col=LEXICON_TOKEN_COL,
            lexicon_score_col=LEXICON_SCORE_COL,
        )

        expected = pl.DataFrame(
            {
                TOKEN_COL: ["the", "cat", "dog", "fish"],
                LEXICON_SCORE_COL: [1.0, 3.0, 7.0, None],
            }
        )

        assert_frame_equal(
            result.select(TOKEN_COL, LEXICON_SCORE_COL), expected, check_row_order=False
        )
        assert "freq" in result.columns


class TestGetImputationExpression:
    @pytest.mark.parametrize(
        "impute, expected_value",
        [(0.0, 0.0), (5.0, 5.0), ("mean", (1.0 + 3.0 + 7.0) / 3), (pl.lit(2.5), 2.5)],
    )
    def test_imputation_value(self, lexicon_dict, impute, expected_value):
        expr = get_imputation_expression(lexicon_dict, impute=impute)

        assert pl.select(expr).item() == expected_value

    def test_mean_of_lexicon_input(self):
        lexicon = Lexicon(
            lf=pl.LazyFrame({"token": ["the", "cat"], "score": [1.0, 3.0]}), center=5.0
        )

        assert pl.select(get_imputation_expression(lexicon, impute="mean")).item() == 2.0

    def test_mean_of_dataframe_input(self, lexicon_df):
        expr = get_imputation_expression(
            lexicon_df, impute="mean", lexicon_score_col=LEXICON_SCORE_COL
        )

        assert pl.select(expr).item() == (1.0 + 3.0 + 7.0) / 3

    def test_mean_of_dataframe_requires_lexicon_score_col(self, lexicon_df):
        with pytest.raises(ValueError, match="lexicon_score_col"):
            get_imputation_expression(lexicon_df, impute="mean")

    def test_invalid_impute_raises(self, lexicon_dict):
        with pytest.raises(ValueError, match="impute"):
            get_imputation_expression(lexicon_dict, impute="median")  # type: ignore[arg-type]


class TestValidation:
    def test_dataframe_lexicon_requires_lexicon_label_col(self, df_words, lexicon_df):
        with pytest.raises(ValueError, match="lexicon_token_col"):
            join_lexicon(df_words, lexicon_df, df_token_col=TOKEN_COL)

    def test_dataframe_lexicon_requires_lexicon_score_col(self, df_words, lexicon_df):
        with pytest.raises(ValueError, match="lexicon_score_col"):
            join_lexicon(
                df_words, lexicon_df, df_token_col=TOKEN_COL, lexicon_token_col=LEXICON_TOKEN_COL
            )

    def test_unsupported_lexicon_type_raises_type_error(self, df_words):
        with pytest.raises(TypeError, match="must be a Mapping or Polars DataFrame/LazyFrame"):
            join_lexicon(df_words, [("the", 1.0)], df_token_col=TOKEN_COL)


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
        result = join_lexicon(df_input, lexicon_dict, df_token_col=TOKEN_COL)
        assert isinstance(result, expected_type)
