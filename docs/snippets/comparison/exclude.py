# --8<-- [start:setup]
import wordlevel as wl

cl = wl.Catalog.from_dict_of_freqs(
    {
        "before": {"the": 30, "good": 12, "bad": 8, "happy": 6, "sad": 4, "calm": 5},
        "after": {"the": 25, "good": 5, "bad": 7, "happy": 3, "sad": 9, "angry": 6},
    }
)
# --8<-- [end:setup]


# --8<-- [start:tokens]
cl = cl.with_comparisons(wl.comp("before", "after").exclude.tokens(["the"]).score.proportion())
# --8<-- [end:tokens]


# --8<-- [start:unshared_tokens]
cl = cl.with_comparisons(
    wl.comp("before", "after").exclude.unshared_tokens().score.kullback_leibler_divergence()
)
# --8<-- [end:unshared_tokens]


# --8<-- [start:by_frequency]
cl = cl.with_comparisons(
    wl.comp("before", "after").exclude.by_frequency(greater_than=20).score.proportion()
)
# --8<-- [end:by_frequency]


# --8<-- [start:by_proportion]
cl = cl.with_comparisons(
    wl.comp("before", "after").exclude.by_proportion(less_than=0.15).score.proportion()
)
# --8<-- [end:by_proportion]


# --8<-- [start:by_lexicon_score]
cl = cl.with_comparisons(
    wl.comp("before", "after")
    .exclude.by_lexicon_score(between=(4.0, 6.0))
    .score.lexicon(wl.lex.labMT("english"))
)
# --8<-- [end:by_lexicon_score]
