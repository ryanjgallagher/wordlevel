# --8<-- [start:setup]
import wordlevel as wl

speeches = wl.Dataset("presidential_speeches")
cl = wl.Catalog(speeches, corpora=["Joe Biden", "Franklin D. Roosevelt"])

cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.rank_divergence(alpha=1/3)
        .alias("roosevelt_biden"),
)
# --8<-- [end:setup]


# --8<-- [start:bar]
chart = cl.plot.bar("roosevelt_biden", max_rank=50)
# --8<-- [end:bar]


# Renders the chart built above as an inline SVG
# --8<-- [start:bar_render]
import vl_convert as vlc

print(vlc.vegalite_to_svg(chart.to_json()))
# --8<-- [end:bar_render]
