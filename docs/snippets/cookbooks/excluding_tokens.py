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


# --8<-- [start:top_tokens]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.rank_divergence(alpha=1/3)
        .alias("roosevelt_biden_rank_divergence"),
)

print(
    cl.comparisons["roosevelt_biden_rank_divergence"].df
    .select("token", "score_rank", "Franklin D. Roosevelt", "Joe Biden")
    .sort("score_rank")
    .head(n=10)
)
# --8<-- [end:top_tokens]


# --8<-- [start:tokens]
stop_words = ["thats", "im", "of", "to", "which", "weve", "you"]

cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .exclude.tokens(stop_words)
        .score.rank_divergence(alpha=1/3)
        .alias("without_stop_words"),
)
# --8<-- [end:tokens]


# --8<-- [start:tokens_effect]
print(
    cl.comparisons["without_stop_words"].df
    .select("token", "score_rank", "Franklin D. Roosevelt", "Joe Biden")
    .sort("score_rank")
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
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.lexicon(wl.lex.labMT("english"))
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


# --8<-- [start:chained_exclusions]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.rank_divergence(alpha=1/3)
        .exclude.tokens(stop_words)
        .exclude.by_frequency(greater_than=40)
        .alias("without_stop_words_or_common_tokens"),
)
# --8<-- [end:chained_exclusions]


# --8<-- [start:chained_exclusions_effect]
print(
    cl.comparisons["without_stop_words_or_common_tokens"].df
    .select("token", "score_rank", "Franklin D. Roosevelt", "Joe Biden")
    .sort("score_rank")
    .head(n=10)
)
# --8<-- [end:chained_exclusions_effect]
