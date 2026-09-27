# Mirrors the setup on the Score Comparisons page: markdown-exec sessions do not cross
# page boundaries, so every page rebuilds its own catalog.
# --8<-- [start:setup]
import wordlevel as wl

speeches = wl.Dataset("presidential_speeches")
cl = wl.Catalog(speeches, corpora=["Joe Biden", "Franklin D. Roosevelt"])
# --8<-- [end:setup]


# --8<-- [start:normalize_total_diff]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.rank_divergence(alpha=1/3)
        .normalize(by="total_diff")
        .alias("roosevelt_biden_rank_divergence_normed"),
)
# --8<-- [end:normalize_total_diff]


# --8<-- [start:normalize_net_diff]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.shannon_entropy()
        .normalize(by="net_diff")
        .alias("roosevelt_biden_entropy_normed"),
)
# --8<-- [end:normalize_net_diff]
