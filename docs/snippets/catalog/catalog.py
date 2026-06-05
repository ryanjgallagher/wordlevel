# --8<-- [start:setup]
import wordlevel as wl

cl = wl.Catalog.from_dict_of_freqs(
    {
        "before": {"good": 12, "bad": 8, "happy": 6, "sad": 4},
        "after": {"good": 5, "bad": 7, "happy": 3, "sad": 9},
    }
)
# --8<-- [end:setup]


# --8<-- [start:from_polars]
import polars as pl
import wordlevel as wl

df = pl.DataFrame(
    {
        "token": ["good", "bad", "happy", "sad"],
        "before": [12, 8, 6, 4],
        "after": [5, 7, 3, 9],
    }
)

cl = wl.Catalog.from_polars(df, token_col="token", corpora=["before", "after"])
# --8<-- [end:from_polars]


# --8<-- [start:from_list_of_freqs]
import wordlevel as wl

cl = wl.Catalog.from_list_of_freqs(
    [
        {"good": 12, "bad": 8, "happy": 6, "sad": 4},
        {"good": 5, "bad": 7, "happy": 3, "sad": 9},
    ],
    corpora=["before", "after"],
)
# --8<-- [end:from_list_of_freqs]


# --8<-- [start:from_dict_of_freqs]
import wordlevel as wl

cl = wl.Catalog.from_dict_of_freqs(
    {
        "before": {"good": 12, "bad": 8, "happy": 6, "sad": 4},
        "after": {"good": 5, "bad": 7, "happy": 3, "sad": 9},
    }
)
# --8<-- [end:from_dict_of_freqs]


# --8<-- [start:from_catalog]
import wordlevel as wl

cl = wl.Catalog.from_dict_of_freqs(
    {
        "before": {"good": 12, "bad": 8, "happy": 6, "sad": 4},
        "after": {"good": 5, "bad": 7, "happy": 3, "sad": 9},
    }
)

new_cl = wl.Catalog.from_catalog(cl)
# --8<-- [end:from_catalog]


# --8<-- [start:with_comparisons]
cl = cl.with_comparisons(
    wl.comp("before", "after").score.kld().alias("kld_comparison"),
    wl.comp("before", "after").score.jsd().alias("jsd_comparison"),
)
# --8<-- [end:with_comparisons]
