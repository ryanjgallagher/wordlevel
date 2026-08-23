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


# --8<-- [start:tokens]
# The corpus strips apostrophes, so contractions appear without them
stop_words = ["thats", "im", "of", "to", "which", "weve", "you"]

cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .exclude.tokens(stop_words)
        .score.rank_divergence(alpha=1/3)
        .normalize(by="sum_abs")
        .alias("without_stop_words"),
)
# --8<-- [end:tokens]


# --8<-- [start:tokens_effect]
print(
    cl.comparisons["without_stop_words"].df
    .sort("score_rank")
    .select("token", "comparison_score", "attribution")
    .head(n=10)
)
# --8<-- [end:tokens_effect]



# --8<-- [start:by_frequency]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .exclude.by_frequency(less_than=10)
        .score.rank_divergence(alpha=1/3)
        .alias("without_rare_tokens"),
)
# --8<-- [end:by_frequency]


# --8<-- [start:by_proportion]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .exclude.by_proportion(greater_than=0.005)
        .score.rank_divergence(alpha=1/3)
        .alias("without_common_tokens"),
)
# --8<-- [end:by_proportion]


# --8<-- [start:by_lexicon_score]
labmt = wl.lex.labMT("english")

cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.lexicon(labmt, reference_score=labmt.center)
        .exclude.by_lexicon_score(between=(4.0, 6.0))
        .alias("without_neutral_words"),
)
# --8<-- [end:by_lexicon_score]


# --8<-- [start:unshared_tokens]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .exclude.unshared_tokens()
        .score.kullback_leibler_divergence()
        .alias("shared_vocabulary_only"),
)
# --8<-- [end:unshared_tokens]
