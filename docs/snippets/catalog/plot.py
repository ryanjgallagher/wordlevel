# --8<-- [start:setup]
import wordlevel as wl

cl = wl.Catalog.from_dict_of_freqs(
    {
        "before": {"good": 12, "bad": 8, "happy": 6, "sad": 4},
        "after": {"good": 5, "bad": 7, "happy": 3, "sad": 9},
    }
).with_comparisons(
    wl.comp("before", "after").score.proportion().alias("basic_comparison"),
    wl.comp("before", "after")
    .score.lexicon(wl.lex.labMT("english"))
    .alias("weighted_avg_comparison"),
)
# --8<-- [end:setup]


# --8<-- [start:bar]
chart = cl.plot.bar("basic_comparison")
# --8<-- [end:bar]


# --8<-- [start:shift]
chart = cl.plot.shift("weighted_avg_comparison")
# --8<-- [end:shift]
