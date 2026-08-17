# --8<-- [start:setup]
import wordlevel as wl

speeches = wl.Dataset("presidential_speeches")
cl = wl.Catalog(speeches, corpora=["Joe Biden", "Franklin D. Roosevelt"])

cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.rank_divergence(alpha=1/3)
        .normalize(by="sum_abs")
        .alias("roosevelt_biden_rank_divergence_normed"),
)
# --8<-- [end:setup]


# Hide printing the shape
# --8<-- [start:display_config]
import polars as pl

pl.Config.set_tbl_hide_dataframe_shape(True)
# --8<-- [end:display_config]


# --8<-- [start:top_tokens]
print(
    cl.comparisons["roosevelt_biden_rank_divergence_normed"].df
    .sort("score_rank")
    .select(
        "token",
        "comparison_score",
        "attribution"
    )
    .head(n=10)
)
# --8<-- [end:top_tokens]
