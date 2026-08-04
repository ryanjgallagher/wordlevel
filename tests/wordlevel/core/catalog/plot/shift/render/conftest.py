import random
from collections.abc import Callable

import polars as pl
import pytest
from polars import col

from wordlevel.constants import SHIFT_SCORE_TYPES
from wordlevel.core.comparison.score.components import (
    frequency_diff,
    higher_score,
    less_of_negative,
    less_of_positive,
    lower_score,
    more_of_negative,
    more_of_positive,
    reference_score_diff,
    weighted_score_diff,
)


negative_score_types = {
    "lower_score",
    "lower_score_counteracted",
    "less_of_positive",
    "less_of_positive_counteracted",
    "more_of_negative",
    "more_of_negative_counteracted",
}


@pytest.fixture
def token_col() -> str:
    return "word"


@pytest.fixture
def rank_col() -> str:
    return "rank"


@pytest.fixture
def score_col() -> str:
    return "contribution"


@pytest.fixture
def score_type_col() -> str:
    return "contribution_type"


@pytest.fixture
def words() -> Callable:
    def _words_factory(n: int = 52) -> list[str]:
        ws = "a b c d e f g h i j k l m n o p q r s t u v w x y z".split()  # ruff: ignore[SIM905]
        ws += [f"{w}{w}" for w in ws]

        return ws[:n]

    return _words_factory


@pytest.fixture()
def make_mock_df(words, token_col, rank_col, score_col, score_type_col) -> Callable:
    """ """

    def _df_factory(n: int = 52, long: bool = True, seed: int = 111) -> pl.DataFrame:
        random.seed(seed)

        ws = words(n=n)
        freqs1 = [random.randint(0, 100) for _ in range(len(ws))]
        freqs2 = [random.randint(0, 100) for _ in range(len(ws))]
        scores1 = [random.uniform(1, 9) for _ in range(len(ws))]
        scores2 = [random.uniform(1, 9) for _ in range(len(ws))]

        component_cols = [col("freq_diff"), col("ref_score_diff"), col("weighted_score_diff")]

        df = (
            pl.DataFrame(
                {
                    token_col: ws,
                    "freqs1": freqs1,
                    "freqs2": freqs2,
                    "scores1": scores1,
                    "scores2": scores2,
                }
            )
            .with_columns(
                frequency_diff(col("freqs1"), col("freqs2")).alias("freq_diff"),
                weighted_score_diff(
                    col("freqs1"), col("freqs2"), col("scores1"), col("scores2")
                ).alias("weighted_score_diff"),
                reference_score_diff(col("scores1"), col("scores2"), reference_score=5).alias(
                    "ref_score_diff"
                ),
            )
            # Get rank
            .with_columns(
                (col("freq_diff") * col("ref_score_diff") + col("weighted_score_diff")).alias(
                    "total_shift"
                ),
            )
            .with_columns(col("total_shift").abs().rank(descending=True).alias(rank_col))
            # Get score components
            .with_columns(
                more_of_positive(*component_cols, counteracted=False).alias("more_of_positive"),
                less_of_negative(*component_cols, counteracted=False).alias("less_of_negative"),
                more_of_negative(*component_cols, counteracted=False).alias("more_of_negative"),
                less_of_positive(*component_cols, counteracted=False).alias("less_of_positive"),
                higher_score(*component_cols, counteracted=False).alias("higher_score"),
                lower_score(*component_cols, counteracted=False).alias("lower_score"),
                more_of_positive(*component_cols, counteracted=True).alias(
                    "more_of_positive_counteracted"
                ),
                less_of_negative(*component_cols, counteracted=True).alias(
                    "less_of_negative_counteracted"
                ),
                more_of_negative(*component_cols, counteracted=True).alias(
                    "more_of_negative_counteracted"
                ),
                less_of_positive(*component_cols, counteracted=True).alias(
                    "less_of_positive_counteracted"
                ),
                higher_score(*component_cols, counteracted=True).alias("higher_score_counteracted"),
                lower_score(*component_cols, counteracted=True).alias("lower_score_counteracted"),
            )
        )

        if long:
            df = df.unpivot(
                on=SHIFT_SCORE_TYPES,
                index=[token_col, rank_col],
                variable_name=score_type_col,
                value_name=score_col,
            )

        return df

    return _df_factory
