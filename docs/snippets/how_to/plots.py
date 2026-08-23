# --8<-- [start:setup]
import wordlevel as wl

speeches = wl.Dataset("presidential_speeches")
cl = wl.Catalog(speeches, corpora=["Joe Biden", "Franklin D. Roosevelt"])

cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.rank_divergence(alpha=1/3)
        .alias("roosevelt_biden"),
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.proportion()
        .alias("roosevelt_biden_proportion"),
)

labmt = wl.lex.labMT("english")
# --8<-- [end:setup]


# --8<-- [start:bar]
chart = cl.plot.bar("roosevelt_biden", max_rank=50)
# --8<-- [end:bar]


# --8<-- [start:max_rank]
chart = cl.plot.bar("roosevelt_biden", max_rank=15)
# --8<-- [end:max_rank]


# --8<-- [start:horizontal]
chart = cl.plot.bar("roosevelt_biden", max_rank=15, vertical=False)
# --8<-- [end:horizontal]


# --8<-- [start:legend_and_size]
chart = cl.plot.bar("roosevelt_biden", max_rank=15, show_legend=True, width=500)
# --8<-- [end:legend_and_size]


# --8<-- [start:rank_size]
from wordlevel.configs import RankAxisConfig

chart = cl.plot.bar(
    "roosevelt_biden",
    max_rank=15,
    width=500,
    rank_axis=RankAxisConfig(bar_width=24),
)
# --8<-- [end:rank_size]


# --8<-- [start:split_nonnegative]
chart = cl.plot.bar("roosevelt_biden", max_rank=15, split_nonnegative_by_corpus=False)
# --8<-- [end:split_nonnegative]


# --8<-- [start:lexicon_comparison]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.lexicon(labmt, reference_score=labmt.center)
        .normalize(by="sum_abs")
        .alias("sentiment"),
)
# --8<-- [end:lexicon_comparison]


# --8<-- [start:shift]
chart = cl.plot.shift("sentiment", max_rank=20)
# --8<-- [end:shift]


# --8<-- [start:bar_totals]
chart = cl.plot.bar("roosevelt_biden", max_rank=15, show_totals=True)
# --8<-- [end:bar_totals]


# --8<-- [start:shift_totals]
chart = cl.plot.shift("sentiment", max_rank=20, show_totals=True)
# --8<-- [end:shift_totals]


# --8<-- [start:entropy_comparison]
cl = cl.with_comparisons(
    wl.comp("Franklin D. Roosevelt", "Joe Biden")
        .score.shannon_entropy(reference_score=0)
        .normalize(by="sum_abs")
        .alias("diversity"),
)

chart = cl.plot.shift("diversity", max_rank=20)
# --8<-- [end:entropy_comparison]


# Renders whichever chart was built most recently as an inline SVG
# --8<-- [start:render]
import vl_convert as vlc

print(vlc.vegalite_to_svg(chart.to_json()))
# --8<-- [end:render]
