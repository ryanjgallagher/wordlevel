# --8<-- [start:setup]
import wordlevel as wl

speeches = wl.Dataset("presidential_speeches")
cl = wl.Catalog(speeches, corpora=["Joe Biden", "Franklin D. Roosevelt"])
# --8<-- [end:setup]


# Hide printing the shape
# --8<-- [start:display_config]
import polars as pl

pl.Config.set_tbl_hide_dataframe_shape(True)
# --8<-- [end:display_config]


# --8<-- [start:proportion]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden").score.proportion().alias("roosevelt_biden")
)
# --8<-- [end:proportion]


# --8<-- [start:inspect]
roosevelt_biden = cl.comparisons["roosevelt_biden"]

print(
    roosevelt_biden.df.select(
        "token",
        "Franklin D. Roosevelt__wl_normed",
        "Joe Biden__wl_normed",
        "comparison_score",
        "score_rank",
        "attribution"
    )
    .sort("score_rank")
    .head(n=10)
)
# --8<-- [end:inspect]


# --8<-- [start:multiple_comparisons]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.rank_divergence(alpha=1/3)
        .alias("roosevelt_biden_rank_divergence"),
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.shannon_entropy()
        .alias("roosevelt_biden_entropy")
)
# --8<-- [end:multiple_comparisons]


# --8<-- [start:comparison_keys]
print(cl.comparisons.keys())
# --8<-- [end:comparison_keys]


# --8<-- [start:normalize_sum_abs]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.rank_divergence(alpha=1/3)
        .normalize(by="sum_abs")
        .alias("roosevelt_biden_rank_divergence_normed"),
)
# --8<-- [end:normalize_sum_abs]


# --8<-- [start:normalize_abs_sum]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.shannon_entropy()
        .normalize(by="abs_sum")
        .alias("roosevelt_biden_entropy_normed"),
)
# --8<-- [end:normalize_abs_sum]


# --8<-- [start:normalization_metadata]
comparison = cl.comparisons["roosevelt_biden_rank_divergence_normed"]

print(f"Comparison is normalized: {comparison.normalized}")
print(f"Normalization method:     {comparison.normalized_by}")
print(f"Normalization constant:   {comparison.normalization_constant}")
# --8<-- [end:normalization_metadata]
