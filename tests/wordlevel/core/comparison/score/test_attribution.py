import polars as pl
import pytest
from polars.testing import assert_series_equal

from wordlevel.core.comparison.score.attribution import (
    attribute_scores,
    frequency_attribution,
    rank_attribution,
    score_sign_attribution,
)


@pytest.mark.parametrize(
    "normalize,expected_attributions",
    [
        (False, ["comparison", "comparison"]),
        (True, ["reference", "comparison"]),
    ],
)
def test_frequency_attribution(normalize, expected_attributions):
    """ """
    # Arrange
    df = pl.DataFrame(
        {
            "ref_col": [1.0, 3.0],  # Normalized: [0.25, 0.75]
            "cmp_col": [2.0, 8.0],  # Normalized: [0.20, 0.80]
        }
    )

    expected = pl.Series(expected_attributions, dtype=pl.Enum(["reference", "comparison"]))

    # Act
    expr = frequency_attribution(
        reference=pl.col("ref_col"), comparison=pl.col("cmp_col"), normalize=normalize
    )
    actual = df.select(expr.alias("attribution"))["attribution"]

    # Assert
    assert_series_equal(actual, expected, check_names=False)


def test_rank_attribution():
    """ """
    # Arrange
    df = pl.DataFrame(
        {
            "ref_col": [5, 4, 3, 2, 1],  # Ranks: [1, 2, 3, 4, 5]
            "cmp_col": [8, 2, 4, 6, 7],  # Ranks: [1, 5, 4, 3, 2]
        }
    )

    expected = pl.Series(
        [
            "comparison",
            "reference",
            "reference",
            "comparison",
            "comparison",
        ],
        dtype=pl.Enum(["reference", "comparison"]),
    )

    # Act
    expr = rank_attribution(reference=pl.col("ref_col"), comparison=pl.col("cmp_col"))
    actual = df.select(expr.alias("attribution"))["attribution"]

    # Assert
    assert_series_equal(actual, expected, check_names=False)


def test_score_sign_attribution():
    """ """
    # Arrange 1: Total shift is positive (sum = 2)
    # Signs: [+, +, -]. Sum Sign: [+].
    # Same sign as the total reinforces it; opposite sign offsets it: [reinforce, reinforce, offset]
    df_pos = pl.DataFrame({"score": [3, 1, -2]})
    expected_pos = pl.Series(
        ["reinforce", "reinforce", "offset"],
        dtype=pl.Enum(["reinforce", "offset"]),
    )

    # Arrange 2: Total shift is negative (sum = -2)
    # Signs: [+, +, -]. Sum Sign: [-].
    # Same sign as the total reinforces it; opposite sign offsets it: [offset, offset, reinforce]
    df_neg = pl.DataFrame({"score": [1, 2, -5]})
    expected_neg = pl.Series(
        ["offset", "offset", "reinforce"],
        dtype=pl.Enum(["reinforce", "offset"]),
    )

    # Act & Assert 1
    expr_pos = score_sign_attribution(score=pl.col("score"))
    actual_pos = df_pos.select(expr_pos.alias("attribution"))["attribution"]
    assert_series_equal(actual_pos, expected_pos, check_names=False)

    # Act & Assert 2
    expr_neg = score_sign_attribution(score=pl.col("score"))
    actual_neg = df_neg.select(expr_neg.alias("attribution"))["attribution"]
    assert_series_equal(actual_neg, expected_neg, check_names=False)


@pytest.mark.parametrize(
    "attribute_by,attribution_func,labels,func_kwargs",
    [
        (
            "frequency",
            frequency_attribution,
            ("attributed_to_ref", "attributed_to_cmp"),
            {
                "reference_attribution": "attributed_to_ref",
                "comparison_attribution": "attributed_to_cmp",
            },
        ),
        (
            "rank",
            rank_attribution,
            ("attributed_to_ref", "attributed_to_cmp"),
            {
                "reference_attribution": "attributed_to_ref",
                "comparison_attribution": "attributed_to_cmp",
            },
        ),
        (
            "score_sign",
            score_sign_attribution,
            ("attributed_to_reinforce", "attributed_to_offset"),
            {
                "reinforce_attribution": "attributed_to_reinforce",
                "offset_attribution": "attributed_to_offset",
            },
        ),
    ],
)
def test_attribute_score(attribute_by, attribution_func, labels, func_kwargs):
    """`attribute_scores` dispatches to the right function and uses the `labels` override."""
    # Arrange
    df = pl.DataFrame(
        {"ref": [1, 6, 4, 3, 2, 1], "cmp": [2, 3, 6, 7, 1, 0], "scr": [-1, 3, 2, 0, 4, 5]}
    )

    if attribute_by in {"frequency", "rank"}:
        params = {"reference": pl.col("ref"), "comparison": pl.col("cmp")}
    else:
        params = {"score": pl.col("scr")}

    df = df.with_columns(attribution_func(**params, **func_kwargs).alias("expected_attribution"))

    # Act
    df = attribute_scores(
        df,
        by=attribute_by,
        reference=pl.col("ref"),
        comparison=pl.col("cmp"),
        score=pl.col("scr"),
        labels=labels,
        attribution_alias="actual_attribution",
    )

    # Assert
    assert_series_equal(df["expected_attribution"], df["actual_attribution"], check_names=False)
