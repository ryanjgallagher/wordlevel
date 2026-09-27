# --8<-- [start:setup]
import wordlevel as wl

cl = wl.Catalog.from_dict_of_freqs(
    {
        "before": {"good": 12, "bad": 8, "happy": 6, "sad": 4},
        "after": {"good": 5, "bad": 7, "happy": 3, "sad": 9},
    }
)
# --8<-- [end:setup]


# --8<-- [start:normalize]
cl = cl.with_comparisons(wl.comp("before", "after").score.shannon_entropy().normalize("total_diff"))
# --8<-- [end:normalize]
