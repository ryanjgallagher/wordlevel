import polars as pl
import polars.testing

from polars import col

from wordlevel.expressions.shift_plot_components import get_shift_plot_components


def test_shift_plot_components_v2():
    """ """
    test_rows = [
        # More of positive + higher score
        (2, 3, 4, 6, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0),
        (1, 5, 8, 5, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0),
        # Less of negative + higher score
        (-2, -4, 7, 0, 8, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0),
        (-3, -6, 2, 0, 18, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0),
        # More of negative + lower score
        (4, -3, -5, 0, 0, -12, 0, 0, -5, 0, 0, 0, 0, 0, 0),
        (3, -5, -6, 0, 0, -15, 0, 0, -6, 0, 0, 0, 0, 0, 0),
        # Less of positive + lower score
        (-7, 6, -2, 0, 0, 0, -42, 0, -2, 0, 0, 0, 0, 0, 0),
        (-6, 9, -8, 0, 0, 0, -54, 0, -8, 0, 0, 0, 0, 0, 0),
        # More of positive > lower score
        (3, 7, -9, 12, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, -9),
        (5, 4, -6, 14, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, -6),
        # More of positive < lower score
        (2, 3, -9, 0, 0, 0, 0, 0, -3, 6, 0, 0, 0, 0, -6),
        (3, 5, -16, 0, 0, 0, 0, 0, -1, 15, 0, 0, 0, 0, -15),
        # Less of negative > lower score
        (-9, -3, -12, 0, 15, 0, 0, 0, 0, 0, 12, 0, 0, 0, -12),
        (-8, -7, -20, 0, 36, 0, 0, 0, 0, 0, 20, 0, 0, 0, -20),
        # Less of negative < lower score
        (-4, -6, -31, 0, 0, 0, 0, 0, -7, 0, 24, 0, 0, 0, -24),
        (-5, -5, -29, 0, 0, 0, 0, 0, -4, 0, 25, 0, 0, 0, -25),
        # More of negative > higher score
        (6, -3, 14, 0, 0, -4, 0, 0, 0, 0, 0, -14, 0, 14, 0),
        (5, -6, 20, 0, 0, -10, 0, 0, 0, 0, 0, -20, 0, 20, 0),
        # More of negative < higher score
        (7, -3, 22, 0, 0, 0, 0, 1, 0, 0, 0, -21, 0, 21, 0),
        (4, -9, 50, 0, 0, 0, 0, 14, 0, 0, 0, -36, 0, 36, 0),
        # Less of positive > higher score
        (-2, 6, 10, 0, 0, 0, -2, 0, 0, 0, 0, 0, -10, 10, 0),
        (-5, 2, 3, 0, 0, 0, -7, 0, 0, 0, 0, 0, -3, 3, 0),
        # Less of positive < higher score
        (-3, 6, 20, 0, 0, 0, 0, 2, 0, 0, 0, 0, -18, 18, 0),
        (-3, 4, 30, 0, 0, 0, 0, 18, 0, 0, 0, 0, -12, 12, 0),
    ]

    df_expected = pl.DataFrame(
        test_rows,
        schema=[
            "frequency_diff",
            "reference_score_diff",
            "weighted_score_diff",
            "more_of_positive",
            "less_of_negative",
            "more_of_negative",
            "less_of_positive",
            "higher_score",
            "lower_score",
            "more_of_positive_counteracted",
            "less_of_negative_counteracted",
            "more_of_negative_counteracted",
            "less_of_positive_counteracted",
            "higher_score_counteracted",
            "lower_score_counteracted",
        ],
        orient="row",
    )

    df_actual = df_expected.select("frequency_diff", "weighted_score_diff", "reference_score_diff")
    df_actual = get_shift_plot_components(
        df_actual,
        frequency_diff=col("frequency_diff"),
        weighted_score_diff=col("weighted_score_diff"),
        reference_score_diff=col("reference_score_diff"),
    )

    polars.testing.assert_frame_equal(df_actual, df_expected, check_column_order=False)
