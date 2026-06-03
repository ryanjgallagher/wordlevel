# --8<-- [start:setup]
import wordlevel as wl

cl = wl.Catalog.from_dict_of_freqs(
    {
        "before": {"good": 12, "bad": 8, "happy": 6, "sad": 4},
        "after": {"good": 5, "bad": 7, "happy": 3, "sad": 9},
    }
)
# --8<-- [end:setup]


# --8<-- [start:frequency]
cl = cl.with_comparisons(wl.comp("before", "after").score.frequency())
# --8<-- [end:frequency]


# --8<-- [start:proportion]
cl = cl.with_comparisons(wl.comp("before", "after").score.proportion())
# --8<-- [end:proportion]


# --8<-- [start:shannon_entropy]
cl = cl.with_comparisons(wl.comp("before", "after").score.shannon_entropy())
# --8<-- [end:shannon_entropy]


# --8<-- [start:tsallis_entropy]
cl = cl.with_comparisons(wl.comp("before", "after").score.tsallis_entropy())
# --8<-- [end:tsallis_entropy]


# --8<-- [start:lexicon]
cl = cl.with_comparisons(wl.comp("before", "after").score.lexicon(wl.lex.labMT("english")))
# --8<-- [end:lexicon]


# --8<-- [start:kullback_leibler_divergence]
cl = cl.with_comparisons(wl.comp("before", "after").score.kullback_leibler_divergence())
# --8<-- [end:kullback_leibler_divergence]


# --8<-- [start:jensen_shannon_divergence]
cl = cl.with_comparisons(wl.comp("before", "after").score.jensen_shannon_divergence())
# --8<-- [end:jensen_shannon_divergence]


# --8<-- [start:rank_divergence]
cl = cl.with_comparisons(wl.comp("before", "after").score.rank_divergence())
# --8<-- [end:rank_divergence]
