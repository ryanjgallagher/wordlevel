import polars as pl
import polars.testing
import pytest
from polars import col
from polars.testing import assert_series_equal, assert_series_not_equal

from wordlevel.constants import IS_BORROWED_COL, SHIFT_SCORE_TYPES
from wordlevel.core.comparison.score.attribution import ATTRIBUTION_LABELS
from wordlevel.core.comparison.score.lexicon import DEFAULT_LEXICON_SCORE_COL
from wordlevel.core.comparison.score.plan import (
    basic_score_plan,
    lexicon_join_plan,
    weighted_average_score_plan,
)
from wordlevel.lexicons.core.load import Lexicon


_TOKEN_COL = "word"
_LEXICON_LABEL_COL = "token"
_LEXICON_SCORE_COL = "sentiment"


@pytest.fixture
def df_words() -> pl.DataFrame:
    return pl.DataFrame(
        {
            _TOKEN_COL: ["the", "cat", "dog", "fish"],
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
            token=col("word"),
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
            token=col("token"),
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
            token=col("word"),
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
                # inverted scores = freqs_ref - freqs_cmp = [-5, 1, 3, -4]; sum = -5
                # same sign as the total reinforces it, opposite offsets it: [reinf, off, off, reinf]
                ["reinforce", "offset", "offset", "reinforce"],
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
            token=col("word"),
            score_func=score_func,
            attribute_by=attribute_by,
            attribution_alias="attribution",
        )

        expected = pl.Series(
            expected_attribution, dtype=pl.Enum(list(ATTRIBUTION_LABELS[attribute_by]))
        )
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
            token=col("token"),
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

    def test_passthrough_cols_survive_the_score_column_select(self) -> None:
        """Only the input columns named in `passthrough_cols` are carried into the output."""
        df = pl.DataFrame(
            {
                "token": ["a", "b"],
                "freqs_ref": [0.4, 0.6],
                "freqs_cmp": [0.3, 0.7],
                "keep_me": [True, False],
                "drop_me": [1, 2],
            }
        )

        df_result = weighted_average_score_plan(
            df,
            score_freqs_ref=col("freqs_ref"),
            score_freqs_cmp=col("freqs_cmp"),
            component_freqs_ref=col("freqs_ref"),
            component_freqs_cmp=col("freqs_cmp"),
            token=col("token"),
            score_func=simple_dual_score_func,
            clean_score_func=simple_clean_score_func,
            attribute_by="frequency",
            passthrough_cols=["keep_me"],
        )

        assert "keep_me" in df_result.columns
        assert "drop_me" not in df_result.columns

    @pytest.mark.parametrize(
        "attribute_by, expected_attribution",
        [
            (
                "frequency",
                # freqs_cmp > freqs_ref -> ["comparison", "comparison"]
                # (total scores are negative with reference_score=10, so score_sign would give
                # ["offset", "offset"] -- proves frequency columns are wired, not score)
                ["comparison", "comparison"],
            ),
            (
                "score_sign",
                # total scores are negative despite freqs_cmp > freqs_ref (reference_score=10
                # dominates); both contributions share the total's sign, so they reinforce it
                ["reinforce", "reinforce"],
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
            token=col("word"),
            score_func=simple_dual_score_func,
            clean_score_func=simple_clean_score_func,
            attribute_by=attribute_by,
            reference_score=10.0,
            attribution_alias="attribution",
        )

        expected = pl.Series(
            expected_attribution, dtype=pl.Enum(list(ATTRIBUTION_LABELS[attribute_by]))
        )
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
            df_token_col=_TOKEN_COL,
            lexicon=lexicon_dict,
            lexicon_score_col=lexicon_score_col,
            score_alias_ref=score_alias_ref,
            score_alias_cmp=score_alias_cmp,
        )

        assert expected_alias_ref in result.columns
        assert expected_alias_cmp in result.columns
        assert _TOKEN_COL in result.columns
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
            df_token_col=_TOKEN_COL,
            lexicon=lexicon,
            score_alias_ref=score_alias_ref,
            score_alias_cmp=score_alias_cmp,
        )

        assert expected_alias_ref in result.columns
        assert expected_alias_cmp in result.columns
        assert _TOKEN_COL in result.columns
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
            df_token_col=_TOKEN_COL,
            lexicon=lexicon_df,
            lexicon_token_col=_LEXICON_LABEL_COL,
            lexicon_score_col=_LEXICON_SCORE_COL,
            score_alias_ref=score_alias_ref,
            score_alias_cmp=score_alias_cmp,
        )

        assert expected_alias_ref in result.columns
        assert expected_alias_cmp in result.columns
        assert _TOKEN_COL in result.columns
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
            df_token_col=_TOKEN_COL,
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
            df_token_col=_TOKEN_COL,
            lexicon_ref=lexicon_dict,
            lexicon_cmp=lexicon_dict_alt,
            score_alias_ref="score_ref",
            score_alias_cmp="score_cmp",
        )

        assert_series_not_equal(result["score_ref"], result["score_cmp"], check_names=False)

    class TestValidation:
        def test_no_lexicon_raises(self, df_words) -> None:
            with pytest.raises(ValueError, match="lexicon"):
                lexicon_join_plan(df_words, df_token_col=_TOKEN_COL)

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
                    df_token_col=_TOKEN_COL,
                    lexicon=lexicon_dict,
                    **{lexicon_keyword: lexicon_dict},
                )

        def test_lexicon_and_lexicon_cmp_raises(self, df_words, lexicon_dict) -> None:
            with pytest.raises(ValueError, match="ambiguous"):
                lexicon_join_plan(
                    df_words,
                    df_token_col=_TOKEN_COL,
                    lexicon=lexicon_dict,
                    lexicon_cmp=lexicon_dict,
                )

    class TestImputation:
        def test_default_drops_words_missing_from_lexicon(self, df_words, lexicon_dict) -> None:
            """By default, words not in the lexicon are dropped."""
            result = lexicon_join_plan(
                df_words,
                df_token_col=_TOKEN_COL,
                lexicon=lexicon_dict,
                score_alias_ref="score_ref",
                score_alias_cmp="score_cmp",
            )

            # "fish" is in df_words but not lexicon_dict
            assert "fish" not in result[_TOKEN_COL].to_list()
            assert result["score_ref"].is_not_null().all()
            assert result["score_cmp"].is_not_null().all()

        def test_drop_null_scores_disabled_keeps_nulls(self, df_words, lexicon_dict) -> None:
            """With drop_null_scores=False, null scores are retained."""
            result = lexicon_join_plan(
                df_words,
                df_token_col=_TOKEN_COL,
                lexicon=lexicon_dict,
                score_alias_ref="score_ref",
                score_alias_cmp="score_cmp",
                drop_null_scores=False,
            )

            assert "fish" in result[_TOKEN_COL].to_list()
            fish_row = result.filter(col(_TOKEN_COL) == "fish")
            assert fish_row["score_ref"].item() is None
            assert fish_row["score_cmp"].item() is None

        def test_borrow_fills_missing_score_from_other_corpus(self) -> None:
            """With scored_in_one_lexicon='borrow', a null in one column is filled from the other and flagged."""
            df = pl.DataFrame({_TOKEN_COL: ["the", "dog", "fish"], "freq": [100, 25, 10]})
            lexicon_ref = {"the": 1.0, "fish": 9.0}  # missing "dog"
            lexicon_cmp = {"the": 2.0, "dog": 5.0}  # missing "fish"

            result = lexicon_join_plan(
                df,
                df_token_col=_TOKEN_COL,
                lexicon_ref=lexicon_ref,
                lexicon_cmp=lexicon_cmp,
                score_alias_ref="score_ref",
                score_alias_cmp="score_cmp",
                scored_in_one_lexicon="borrow",
                drop_null_scores=False,
            )

            assert set(result[_TOKEN_COL].to_list()) == {"the", "dog", "fish"}

            # "dog" missing from ref, borrowed from cmp
            dog_row = result.filter(col(_TOKEN_COL) == "dog")
            assert dog_row["score_ref"].item() == 5.0
            assert dog_row["score_cmp"].item() == 5.0
            assert dog_row[IS_BORROWED_COL].item()

            # "fish" missing from cmp, borrowed from ref
            fish_row = result.filter(col(_TOKEN_COL) == "fish")
            assert fish_row["score_ref"].item() == 9.0
            assert fish_row["score_cmp"].item() == 9.0
            assert fish_row[IS_BORROWED_COL].item()

            # "the" present in both lexicons - each keeps its own value, nothing is borrowed
            the_row = result.filter(col(_TOKEN_COL) == "the")
            assert the_row["score_ref"].item() == 1.0
            assert the_row["score_cmp"].item() == 2.0
            assert not the_row[IS_BORROWED_COL].item()

        def test_borrow_with_drop_null_scores_removes_words_missing_from_both_lexicons(
            self,
        ) -> None:
            """Borrow plus drop_null_scores=True removes words missing from both lexicons."""
            df = pl.DataFrame({_TOKEN_COL: ["the", "fish", "tree"], "freq": [100, 10, 5]})
            lexicon_ref = {"the": 1.0}  # missing "fish" and "tree"
            lexicon_cmp = {"the": 2.0, "fish": 9.0}  # missing "tree"

            result = lexicon_join_plan(
                df,
                df_token_col=_TOKEN_COL,
                lexicon_ref=lexicon_ref,
                lexicon_cmp=lexicon_cmp,
                score_alias_ref="score_ref",
                score_alias_cmp="score_cmp",
                scored_in_one_lexicon="borrow",
                drop_null_scores=True,
            )

            # "tree" missing from both lexicons - dropped
            assert "tree" not in result[_TOKEN_COL].to_list()
            # "fish" missing from ref, borrowed from cmp
            fish_row = result.filter(col(_TOKEN_COL) == "fish")
            assert fish_row["score_ref"].item() == 9.0
            assert fish_row["score_cmp"].item() == 9.0

        def test_drop_null_scores_removes_rows_when_only_one_score_is_null(
            self,
        ) -> None:
            """drop_null_scores=True drops rows even when only one of the score columns is null."""
            df = pl.DataFrame({_TOKEN_COL: ["the", "cat"], "freq": [100, 50]})
            lexicon_ref = {"the": 1.0, "cat": 3.0}  # has both words
            lexicon_cmp = {"the": 2.0}  # missing "cat"

            result = lexicon_join_plan(
                df,
                df_token_col=_TOKEN_COL,
                lexicon_ref=lexicon_ref,
                lexicon_cmp=lexicon_cmp,
                score_alias_ref="score_ref",
                score_alias_cmp="score_cmp",
                drop_null_scores=True,
            )

            # "cat" has a non-null score_ref but a null score_cmp -- should still be dropped
            assert result[_TOKEN_COL].to_list() == ["the"]

        def test_drop_keeps_only_words_scored_by_both_lexicons(self) -> None:
            """With "drop", a word scored by only one lexicon is removed rather than filled."""
            df = pl.DataFrame({_TOKEN_COL: ["the", "dog", "fish"], "freq": [100, 25, 10]})

            result = lexicon_join_plan(
                df,
                df_token_col=_TOKEN_COL,
                lexicon_ref={"the": 1.0, "fish": 9.0},  # missing "dog"
                lexicon_cmp={"the": 2.0, "dog": 5.0},  # missing "fish"
                scored_in_one_lexicon="drop",
            )

            assert result[_TOKEN_COL].to_list() == ["the"]
            assert IS_BORROWED_COL not in result.columns

        def test_mean_fills_each_side_from_its_own_lexicon(self) -> None:
            """A token scored by one lexicon gets the mean of the lexicon that lacks it."""
            df = pl.DataFrame({_TOKEN_COL: ["the", "dog", "fish"], "freq": [100, 25, 10]})
            lexicon_ref = {"the": 1.0, "fish": 9.0}  # missing "dog", mean 5.0
            lexicon_cmp = {"the": 2.0, "dog": 4.0}  # missing "fish", mean 3.0

            result = lexicon_join_plan(
                df,
                df_token_col=_TOKEN_COL,
                lexicon_ref=lexicon_ref,
                lexicon_cmp=lexicon_cmp,
                score_alias_ref="score_ref",
                score_alias_cmp="score_cmp",
                scored_in_one_lexicon="mean",
            )

            dog_row = result.filter(col(_TOKEN_COL) == "dog")
            assert (dog_row["score_ref"].item(), dog_row["score_cmp"].item()) == (5.0, 4.0)
            fish_row = result.filter(col(_TOKEN_COL) == "fish")
            assert (fish_row["score_ref"].item(), fish_row["score_cmp"].item()) == (9.0, 3.0)

        @pytest.mark.parametrize("drop_null_scores", [True, False])
        def test_words_missing_from_both_lexicons_are_not_imputed(
            self, drop_null_scores: bool
        ) -> None:
            """Imputation only fills words that one lexicon scores; words in neither stay null."""
            df = pl.DataFrame({_TOKEN_COL: ["the", "tree"], "freq": [100, 5]})

            result = lexicon_join_plan(
                df,
                df_token_col=_TOKEN_COL,
                lexicon_ref={"the": 1.0},
                lexicon_cmp={"the": 2.0},
                score_alias_ref="score_ref",
                score_alias_cmp="score_cmp",
                scored_in_one_lexicon=0.0,
                drop_null_scores=drop_null_scores,
            )

            tree_row = result.filter(col(_TOKEN_COL) == "tree")
            if drop_null_scores:
                assert tree_row.height == 0
            else:
                assert tree_row["score_ref"].item() is None
                assert tree_row["score_cmp"].item() is None

        def test_single_lexicon_imputes_nothing(self, df_words, lexicon_dict) -> None:
            """With one lexicon no word is scored on only one side, so imputing changes nothing."""
            imputed = lexicon_join_plan(
                df_words,
                df_token_col=_TOKEN_COL,
                lexicon=lexicon_dict,
                scored_in_one_lexicon="mean",
            )
            not_imputed = lexicon_join_plan(df_words, df_token_col=_TOKEN_COL, lexicon=lexicon_dict)

            polars.testing.assert_frame_equal(imputed, not_imputed)
